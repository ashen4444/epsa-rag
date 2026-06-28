"""Rule-based next-hop query generation for EPSA.

The generator creates one focused retrieval query only when the current
SufficiencyDecision says evidence is incomplete.

Important boundary:
    This module does not retrieve documents, decide sufficiency, prune context,
    generate final answers, or call an LLM.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from typing import Any

from epsa_rag.epsa.schemas import (
    EvidenceGraph,
    EvidencePath,
    NextHopQuery,
    QuestionAnalysis,
    SufficiencyDecision,
)


_ANSWER_TYPE_KEYWORDS = {
    "LOCATION": "location",
    "DATE": "date",
    "NUMBER": "number",
    "PERSON": "person",
    "ORGANIZATION": "organization",
    "TITLE_OR_WORK": "title",
    "ENTITY": "entity",
    "BOOLEAN": "evidence",
    "UNKNOWN": "evidence",
}

_RELATION_QUERY_TERMS = {
    "born": "born birthplace",
    "birthplace": "born birthplace",
    "directed": "directed director",
    "written": "written author",
    "author": "author written",
    "located": "located location",
    "capital": "capital location",
    "population": "population number",
    "length": "length",
    "released": "released date",
    "published": "published date",
    "founded": "founded founder",
    "starring": "starring cast",
    "member": "member",
    "genre": "genre",
    "occupation": "occupation profession",
    "spouse": "spouse married",
    "parent": "parent",
    "child": "child",
    "educated": "educated alma mater",
    "discovered": "discovered discovery",
    "capacity": "capacity seats",
}

_GENERIC_MISSING_PHRASES = {
    "no candidate evidence path found",
    "no complete bridge evidence path found",
    "no complete factoid evidence path found",
    "no connected yes/no evidence path found",
}


class NextHopQueryGenerator:
    """Generate deterministic next-hop retrieval queries from missing evidence."""

    def generate(
        self,
        question_analysis: QuestionAnalysis,
        sufficiency_decision: SufficiencyDecision,
        evidence_graph: EvidenceGraph | None = None,
        evidence_paths: list[EvidencePath] | None = None,
    ) -> NextHopQuery:
        """Return one focused next-hop query, or a no-query result.

        Args:
            question_analysis: Structured deterministic question analysis.
            sufficiency_decision: Output from SufficiencyDecisionEngine.
            evidence_graph: Optional graph, used only for metadata/context.
            evidence_paths: Optional candidate paths for fallback partial-path selection.

        Returns:
            NextHopQuery. ``query`` is None when no useful deterministic query can
            be generated.
        """

        if sufficiency_decision.sufficient:
            return self._no_query(
                reason="Evidence is already sufficient; no next-hop query is needed.",
                source="sufficiency_decision",
                metadata={
                    "sufficient": True,
                    "calls_llm": False,
                    "retrieves_documents": False,
                    "makes_sufficiency_decision": False,
                },
            )

        question_type = _question_type(question_analysis, sufficiency_decision)
        expected_answer_type = _expected_answer_type(question_analysis, sufficiency_decision)
        best_path = sufficiency_decision.best_path or _best_fallback_path(evidence_paths or [])
        seed_entities = _seed_entities(question_analysis)
        comparison_targets = _comparison_targets(question_analysis)
        required_relations = _required_relations(question_analysis)
        path_relations = _path_relations(best_path)
        missing_relation = _choose_missing_relation(
            missing_evidence=sufficiency_decision.missing_evidence,
            required_relations=required_relations,
            path_relations=path_relations,
        )

        if question_type == "comparison":
            return self._comparison_query(
                comparison_targets=comparison_targets,
                seed_entities=seed_entities,
                missing_relation=missing_relation,
                expected_answer_type=expected_answer_type,
                sufficiency_decision=sufficiency_decision,
            )

        if question_type in {"yes_no", "yes-no", "boolean"}:
            return self._yes_no_query(
                seed_entities=seed_entities,
                missing_relation=missing_relation,
                sufficiency_decision=sufficiency_decision,
            )

        target_entity = _choose_target_entity(
            best_path=best_path,
            seed_entities=seed_entities,
            answer_candidate=sufficiency_decision.answer_candidate,
            question_type=question_type,
        )

        if not target_entity and seed_entities:
            target_entity = seed_entities[0]

        if question_type == "bridge":
            query_type = "bridge_completion"
        elif missing_relation:
            query_type = "relation_completion"
        elif expected_answer_type not in {"", "UNKNOWN", "ENTITY"}:
            query_type = "answer_type_completion"
        else:
            query_type = "factoid_completion"

        query = _build_query(
            entities=[target_entity] if target_entity else seed_entities[:1],
            relation=missing_relation,
            expected_answer_type=expected_answer_type,
        )

        if not query:
            return self._no_query(
                reason="No seed, bridge, relation, or expected-answer-type signal was available for deterministic query generation.",
                source="question_analysis+sufficiency_decision",
                metadata=self._metadata(
                    sufficiency_decision=sufficiency_decision,
                    evidence_graph=evidence_graph,
                    best_path=best_path,
                ),
            )

        confidence = _query_confidence(
            has_entity=bool(target_entity or seed_entities),
            has_relation=bool(missing_relation),
            has_answer_type=expected_answer_type not in {"", "UNKNOWN"},
            has_path=best_path is not None,
        )

        return NextHopQuery(
            query=query,
            query_type=query_type,
            source="question_analysis+sufficiency_decision",
            target_entity=target_entity,
            missing_relation=missing_relation,
            expected_answer_type=expected_answer_type or None,
            reason=_reason_text(sufficiency_decision),
            confidence=confidence,
            metadata=self._metadata(
                sufficiency_decision=sufficiency_decision,
                evidence_graph=evidence_graph,
                best_path=best_path,
                extra={
                    "seed_entities": seed_entities,
                    "required_relations": required_relations,
                    "path_relations": path_relations,
                    "calls_llm": False,
                    "retrieves_documents": False,
                    "makes_sufficiency_decision": False,
                },
            ),
        )

    def _comparison_query(
        self,
        *,
        comparison_targets: list[str],
        seed_entities: list[str],
        missing_relation: str | None,
        expected_answer_type: str,
        sufficiency_decision: SufficiencyDecision,
    ) -> NextHopQuery:
        targets = comparison_targets or seed_entities[:2]
        relation = missing_relation or _first_relation_from_text(
            sufficiency_decision.missing_evidence or ""
        )
        query = _build_query(
            entities=targets,
            relation=relation,
            expected_answer_type=expected_answer_type,
        )
        if not query:
            return self._no_query(
                reason="Comparison evidence is incomplete, but no comparison target or relation was available.",
                source="question_analysis+sufficiency_decision",
            )
        return NextHopQuery(
            query=query,
            query_type="comparison_target_completion",
            source="question_analysis+sufficiency_decision",
            target_entity=" | ".join(targets) if targets else None,
            missing_relation=relation,
            expected_answer_type=expected_answer_type or None,
            reason=_reason_text(sufficiency_decision),
            confidence=_query_confidence(
                has_entity=bool(targets),
                has_relation=bool(relation),
                has_answer_type=expected_answer_type not in {"", "UNKNOWN"},
                has_path=sufficiency_decision.best_path is not None,
            ),
            metadata={
                "comparison_targets": targets,
                "calls_llm": False,
                "retrieves_documents": False,
                "makes_sufficiency_decision": False,
            },
        )

    def _yes_no_query(
        self,
        *,
        seed_entities: list[str],
        missing_relation: str | None,
        sufficiency_decision: SufficiencyDecision,
    ) -> NextHopQuery:
        query = _build_query(
            entities=seed_entities,
            relation=missing_relation,
            expected_answer_type="BOOLEAN",
            include_answer_type=False,
        )
        if not query:
            return self._no_query(
                reason="Yes/no evidence is incomplete, but no entity or relation signal was available.",
                source="question_analysis+sufficiency_decision",
            )
        return NextHopQuery(
            query=query,
            query_type="yes_no_relation_check",
            source="question_analysis+sufficiency_decision",
            target_entity=" | ".join(seed_entities) if seed_entities else None,
            missing_relation=missing_relation,
            expected_answer_type="BOOLEAN",
            reason=_reason_text(sufficiency_decision),
            confidence=_query_confidence(
                has_entity=bool(seed_entities),
                has_relation=bool(missing_relation),
                has_answer_type=True,
                has_path=sufficiency_decision.best_path is not None,
            ),
            metadata={
                "seed_entities": seed_entities,
                "calls_llm": False,
                "retrieves_documents": False,
                "makes_sufficiency_decision": False,
            },
        )

    @staticmethod
    def _no_query(
        *,
        reason: str,
        source: str,
        metadata: dict[str, Any] | None = None,
    ) -> NextHopQuery:
        return NextHopQuery(
            query=None,
            query_type="no_query",
            source=source,
            target_entity=None,
            missing_relation=None,
            expected_answer_type=None,
            reason=reason,
            confidence=0.0,
            metadata={
                "calls_llm": False,
                "retrieves_documents": False,
                "makes_sufficiency_decision": False,
                **(metadata or {}),
            },
        )

    @staticmethod
    def _metadata(
        *,
        sufficiency_decision: SufficiencyDecision,
        evidence_graph: EvidenceGraph | None,
        best_path: EvidencePath | None,
        extra: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        metadata = {
            "sufficient": sufficiency_decision.sufficient,
            "decision_reason": sufficiency_decision.decision_reason,
            "missing_evidence": sufficiency_decision.missing_evidence,
            "best_path_id": best_path.path_id if best_path is not None else None,
            "num_graph_nodes": len(evidence_graph.nodes) if evidence_graph is not None else None,
            "num_graph_edges": len(evidence_graph.edges) if evidence_graph is not None else None,
        }
        metadata.update(extra or {})
        return metadata


def _question_type(
    question_analysis: QuestionAnalysis,
    sufficiency_decision: SufficiencyDecision,
) -> str:
    value = getattr(question_analysis, "question_type", None) or sufficiency_decision.question_type
    normalized = _norm_text(value).replace("-", "_")
    if normalized in {"yes no", "yes_no", "boolean"}:
        return "yes_no"
    return normalized or "factoid"


def _expected_answer_type(
    question_analysis: QuestionAnalysis,
    sufficiency_decision: SufficiencyDecision,
) -> str:
    value = getattr(question_analysis, "expected_answer_type", None) or sufficiency_decision.answer_type
    text = _as_text(value).upper().replace(" ", "_").replace("-", "_")
    return text or "UNKNOWN"


def _seed_entities(question_analysis: QuestionAnalysis) -> list[str]:
    return _dedupe_preserve_order(_as_text_list(getattr(question_analysis, "seed_entities", [])))


def _comparison_targets(question_analysis: QuestionAnalysis) -> list[str]:
    return _dedupe_preserve_order(_as_text_list(getattr(question_analysis, "comparison_targets", [])))


def _required_relations(question_analysis: QuestionAnalysis) -> list[str]:
    values = getattr(question_analysis, "required_relation_hints", []) or []
    if isinstance(values, str):
        return [values] if values else []
    try:
        ordered_values = list(values)
    except TypeError:
        ordered_values = [values]
    if any(getattr(value, "start_char", None) is not None for value in ordered_values):
        ordered_values = sorted(
            ordered_values,
            key=lambda value: (
                getattr(value, "start_char", None) is None,
                getattr(value, "start_char", 10**9) if getattr(value, "start_char", None) is not None else 10**9,
                _as_text(value),
            ),
        )
    return _dedupe_preserve_order(_as_text_list(ordered_values))


def _path_relations(path: EvidencePath | None) -> list[str]:
    if path is None:
        return []
    return _dedupe_preserve_order(_as_text_list(getattr(path, "relation_chain", [])))


def _choose_missing_relation(
    *,
    missing_evidence: str | None,
    required_relations: list[str],
    path_relations: list[str],
) -> str | None:
    from_missing = _first_relation_from_text(missing_evidence or "")
    if from_missing:
        return from_missing

    for relation in required_relations:
        if not _relation_matches_any(relation, path_relations):
            return relation

    if required_relations:
        return required_relations[-1]
    return None


def _first_relation_from_text(text: str) -> str | None:
    normalized = _norm_text(text)
    if not normalized:
        return None

    relation_match = re.search(r"required relation ([a-z0-9_ -]+?)(?:\.|$)", normalized)
    if relation_match:
        return relation_match.group(1).strip()

    for relation in _RELATION_QUERY_TERMS:
        relation_norm = _norm_text(relation)
        if re.search(rf"\b{re.escape(relation_norm)}\b", normalized):
            return relation
    return None


def _choose_target_entity(
    *,
    best_path: EvidencePath | None,
    seed_entities: list[str],
    answer_candidate: str | None,
    question_type: str,
) -> str | None:
    if best_path is None:
        return seed_entities[0] if seed_entities else None

    metadata_bridge = _as_text(best_path.metadata.get("bridge_entity"))
    seed_norms = {_norm_text(seed) for seed in seed_entities}
    answer_norm = _norm_text(answer_candidate or best_path.answer_candidate or "")

    if metadata_bridge and _norm_text(metadata_bridge) not in seed_norms:
        return metadata_bridge

    entity_chain = _as_text_list(getattr(best_path, "entity_chain", []))
    if question_type == "bridge":
        for entity in reversed(entity_chain):
            entity_norm = _norm_text(entity)
            if not entity_norm or entity_norm in seed_norms or entity_norm == answer_norm:
                continue
            return entity

    for entity in reversed(entity_chain):
        entity_norm = _norm_text(entity)
        if not entity_norm or entity_norm == answer_norm:
            continue
        return entity

    return seed_entities[0] if seed_entities else None


def _build_query(
    *,
    entities: list[str],
    relation: str | None,
    expected_answer_type: str,
    include_answer_type: bool = True,
) -> str | None:
    parts: list[str] = []
    for entity in entities:
        _append_unique(parts, entity)

    relation_terms = _relation_terms(relation)
    for term in relation_terms:
        _append_unique(parts, term)

    answer_keyword = _ANSWER_TYPE_KEYWORDS.get(expected_answer_type, "")
    if include_answer_type and answer_keyword and answer_keyword != "evidence":
        already_covered = any(_norm_text(answer_keyword) in _norm_text(part) for part in parts)
        semantically_covered = _answer_type_covered_by_relation_terms(
            expected_answer_type=expected_answer_type,
            relation_terms=relation_terms,
        )
        if not already_covered and not semantically_covered:
            _append_unique(parts, answer_keyword)

    clean_parts = [part for part in parts if part and _norm_text(part) not in _GENERIC_MISSING_PHRASES]
    query = " ".join(clean_parts)
    query = re.sub(r"\s+", " ", query).strip()
    return query or None


def _relation_terms(relation: str | None) -> list[str]:
    if not relation:
        return []
    relation_norm = _norm_text(relation)
    mapped = _RELATION_QUERY_TERMS.get(relation_norm, relation)
    return _dedupe_preserve_order(mapped.split())


def _answer_type_covered_by_relation_terms(
    *,
    expected_answer_type: str,
    relation_terms: list[str],
) -> bool:
    relation_term_norms = {_norm_text(term) for term in relation_terms}
    if expected_answer_type == "LOCATION":
        return bool(relation_term_norms.intersection({"birthplace", "capital", "located", "location"}))
    if expected_answer_type == "DATE":
        return "date" in relation_term_norms
    if expected_answer_type == "NUMBER":
        return "number" in relation_term_norms
    if expected_answer_type == "PERSON":
        return "person" in relation_term_norms
    return False


def _query_confidence(
    *,
    has_entity: bool,
    has_relation: bool,
    has_answer_type: bool,
    has_path: bool,
) -> float:
    score = 0.20
    if has_entity:
        score += 0.30
    if has_relation:
        score += 0.25
    if has_answer_type:
        score += 0.10
    if has_path:
        score += 0.10
    return round(min(score, 0.90), 6)


def _reason_text(sufficiency_decision: SufficiencyDecision) -> str:
    if sufficiency_decision.missing_evidence:
        return f"Generated from missing evidence: {sufficiency_decision.missing_evidence}"
    return f"Generated from insufficient decision: {sufficiency_decision.decision_reason}"


def _best_fallback_path(paths: list[EvidencePath]) -> EvidencePath | None:
    if not paths:
        return None
    return sorted(
        paths,
        key=lambda path: (
            -_safe_float(path.score),
            len(path.evidence_unit_ids),
            path.answer_candidate or "",
            path.path_id,
        ),
    )[0]


def _relation_matches_any(required: str, path_relations: list[str]) -> bool:
    required_norm = _norm_text(required)
    for relation in path_relations:
        relation_norm = _norm_text(relation)
        if not relation_norm:
            continue
        if required_norm == relation_norm:
            return True
        if required_norm in relation_norm or relation_norm in required_norm:
            return True
    return False


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
    if not isinstance(values, Iterable):
        return [_as_text(values)]
    return [_as_text(value) for value in values if _as_text(value)]


def _norm_text(value: Any) -> str:
    return " ".join(_as_text(value).lower().replace("_", " ").replace("-", " ").split())


def _safe_float(value: Any) -> float:
    try:
        if value is None:
            return 0.0
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _append_unique(values: list[str], value: str) -> None:
    cleaned = str(value or "").strip()
    if cleaned and cleaned not in values:
        values.append(cleaned)


def _dedupe_preserve_order(values: Iterable[str]) -> list[str]:
    unique: list[str] = []
    seen: set[str] = set()
    for value in values:
        cleaned = str(value or "").strip()
        key = _norm_text(cleaned)
        if not key or key in seen:
            continue
        seen.add(key)
        unique.append(cleaned)
    return unique


__all__ = ["NextHopQueryGenerator"]
