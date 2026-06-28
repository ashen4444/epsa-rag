from __future__ import annotations

import re
from typing import Any, Iterable

from epsa_rag.epsa.schemas import EvidenceUnit, ScoredEvidenceUnit


_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "in",
    "into", "is", "it", "its", "of", "on", "or", "that", "the", "their",
    "then", "there", "this", "to", "was", "were", "which", "who", "whom",
    "whose", "with", "what", "when", "where", "why", "how", "did", "does",
    "do", "had", "has", "have",
}


def _get_value(obj: Any, key: str, default: Any = None) -> Any:
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _normalize_text(value: Any) -> str:
    if value is None:
        return ""

    value = getattr(value, "value", value)
    value = getattr(value, "name", value)

    if hasattr(value, "relation"):
        return str(getattr(value, "relation") or "").strip()

    if hasattr(value, "answer_type"):
        return _normalize_text(getattr(value, "answer_type"))

    if hasattr(value, "text"):
        return str(getattr(value, "text") or "").strip()

    return str(value or "").strip()


def _normalize_label(value: Any) -> str:
    return _normalize_text(value).upper().replace(" ", "_").replace("-", "_")


def _normalize_set(values: Iterable[Any]) -> set[str]:
    normalized_values: set[str] = set()

    for value in values:
        normalized = _normalize_text(value).casefold().strip()
        if normalized:
            normalized_values.add(normalized)

    return normalized_values


def _tokenize(text: str) -> set[str]:
    return {
        token
        for token in re.findall(r"[a-z0-9]+", text.casefold())
        if token not in _STOPWORDS and len(token) > 1
    }


def _safe_ratio(numerator: int, denominator: int) -> float:
    if denominator <= 0:
        return 0.0
    return numerator / denominator


def _clip(value: float, lower: float = 0.0, upper: float = 1.0) -> float:
    return max(lower, min(upper, value))


class EvidenceScorer:
    """
    Scores sentence-level evidence units using deterministic evidence-quality features.

    This module does not make final sufficiency decisions.
    """

    def score(
        self,
        evidence_unit: EvidenceUnit,
        question_analysis: Any,
        candidate_evidence: Any | None = None,
    ) -> ScoredEvidenceUnit:
        seed_entities = list(_get_value(question_analysis, "seed_entities", []) or [])
        required_relations = list(_get_value(question_analysis, "required_relation_hints", []) or [])

        expected_answer_type = _normalize_label(
            _get_value(question_analysis, "expected_answer_type", "UNKNOWN")
        )

        question_text = _normalize_text(
            _get_value(question_analysis, "normalized_question")
            or _get_value(question_analysis, "question")
            or " ".join(_normalize_text(seed) for seed in seed_entities)
        )

        unit_entities = list(evidence_unit.entities or [])
        unit_relations = list(evidence_unit.relation_hints or [])
        unit_answer_types = [
            _normalize_label(value)
            for value in evidence_unit.answer_type_candidates or []
        ]

        entity_match_score = self._entity_match_score(
            seed_entities,
            unit_entities,
            evidence_unit.resolved_text,
        )
        relation_match_score = self._relation_match_score(
            required_relations,
            unit_relations,
        )
        answer_type_match_score = self._answer_type_match_score(
            expected_answer_type,
            unit_answer_types,
        )
        token_overlap_score = self._token_overlap_score(
            question_text,
            evidence_unit.resolved_text,
        )
        title_match_score = self._title_match_score(
            evidence_unit.doc_title,
            seed_entities,
            question_text,
        )
        retrieval_score_component = self._retrieval_component(
            evidence_unit,
            candidate_evidence,
        )
        bridge_entity_score = self._bridge_entity_score(
            seed_entities,
            unit_entities,
            evidence_unit.doc_title,
            unit_relations,
        )
        noise_penalty = self._noise_penalty(
            evidence_unit,
            unit_entities,
            unit_relations,
        )

        weighted_score = (
            0.25 * entity_match_score
            + 0.20 * relation_match_score
            + 0.15 * answer_type_match_score
            + 0.15 * token_overlap_score
            + 0.10 * title_match_score
            + 0.05 * retrieval_score_component
            + 0.10 * bridge_entity_score
            - noise_penalty
        )

        final_score = round(_clip(weighted_score), 6)

        breakdown = {
            "entity_match_score": round(entity_match_score, 6),
            "relation_match_score": round(relation_match_score, 6),
            "answer_type_match_score": round(answer_type_match_score, 6),
            "token_overlap_score": round(token_overlap_score, 6),
            "title_match_score": round(title_match_score, 6),
            "retrieval_score_component": round(retrieval_score_component, 6),
            "bridge_entity_score": round(bridge_entity_score, 6),
            "noise_penalty": round(noise_penalty, 6),
        }

        return ScoredEvidenceUnit(
            evidence_unit=evidence_unit,
            final_score=final_score,
            score_breakdown=breakdown,
        )

    def score_many(
        self,
        evidence_units: Iterable[EvidenceUnit],
        question_analysis: Any,
    ) -> list[ScoredEvidenceUnit]:
        return [
            self.score(evidence_unit=evidence_unit, question_analysis=question_analysis)
            for evidence_unit in evidence_units
        ]

    def rank(
        self,
        evidence_units: Iterable[EvidenceUnit],
        question_analysis: Any,
    ) -> list[ScoredEvidenceUnit]:
        scored_units = self.score_many(evidence_units, question_analysis)
        return sorted(scored_units, key=lambda item: item.final_score, reverse=True)

    @staticmethod
    def _entity_match_score(
        seed_entities: list[Any],
        unit_entities: list[str],
        resolved_text: str,
    ) -> float:
        if not seed_entities:
            return 0.0

        text_lower = resolved_text.casefold()
        unit_entity_set = _normalize_set(unit_entities)

        matches = 0
        for seed in seed_entities:
            normalized_seed = _normalize_text(seed).casefold().strip()
            if not normalized_seed:
                continue

            if normalized_seed in unit_entity_set or normalized_seed in text_lower:
                matches += 1

        return _clip(_safe_ratio(matches, len(seed_entities)))

    @staticmethod
    def _relation_match_score(
        required_relations: list[Any],
        unit_relations: list[str],
    ) -> float:
        if not required_relations:
            return 0.0

        required = _normalize_set(required_relations)
        present = _normalize_set(unit_relations)

        matches = len(required.intersection(present))
        return _clip(_safe_ratio(matches, len(required)))

    @staticmethod
    def _answer_type_match_score(
        expected_answer_type: str,
        unit_answer_types: list[str],
    ) -> float:
        if not expected_answer_type or expected_answer_type == "UNKNOWN":
            return 0.0

        if expected_answer_type in unit_answer_types:
            return 1.0

        if expected_answer_type == "ENTITY" and unit_answer_types:
            return 0.6

        return 0.0

    @staticmethod
    def _token_overlap_score(question_text: str, resolved_text: str) -> float:
        question_tokens = _tokenize(question_text)
        sentence_tokens = _tokenize(resolved_text)

        if not question_tokens:
            return 0.0

        return _clip(len(question_tokens.intersection(sentence_tokens)) / len(question_tokens))

    @staticmethod
    def _title_match_score(
        doc_title: str,
        seed_entities: list[Any],
        question_text: str,
    ) -> float:
        if not doc_title:
            return 0.0

        title = doc_title.casefold().strip()

        if any(title == _normalize_text(seed).casefold().strip() for seed in seed_entities):
            return 1.0

        if title and title in question_text.casefold():
            return 0.8

        return 0.0

    @staticmethod
    def _retrieval_component(
        evidence_unit: EvidenceUnit,
        candidate_evidence: Any | None,
    ) -> float:
        rank = evidence_unit.retrieval_rank

        if rank is None and candidate_evidence is not None:
            rank = _get_value(candidate_evidence, "retrieval_rank", None)

        if rank is not None:
            try:
                rank_int = int(rank)
                if rank_int > 0:
                    return _clip(1.0 / rank_int)
            except (TypeError, ValueError):
                pass

        score = evidence_unit.retrieval_score

        if score is None and candidate_evidence is not None:
            score = _get_value(candidate_evidence, "retrieval_score", None)

        if score is None:
            return 0.0

        try:
            return _clip(float(score))
        except (TypeError, ValueError):
            return 0.0

    @staticmethod
    def _bridge_entity_score(
        seed_entities: list[Any],
        unit_entities: list[str],
        doc_title: str,
        unit_relations: list[str],
    ) -> float:
        seeds = _normalize_set(seed_entities)
        title = doc_title.casefold().strip()

        bridge_entities = [
            entity
            for entity in unit_entities
            if entity.casefold().strip() not in seeds
            and entity.casefold().strip() != title
        ]

        if not bridge_entities:
            return 0.0

        if unit_relations:
            return 1.0

        return 0.5

    @staticmethod
    def _noise_penalty(
        evidence_unit: EvidenceUnit,
        unit_entities: list[str],
        unit_relations: list[str],
    ) -> float:
        tokens = _tokenize(evidence_unit.resolved_text)
        penalty = 0.0

        if len(tokens) < 4:
            penalty += 0.20

        if len(tokens) > 60:
            penalty += 0.10

        if not unit_entities and not unit_relations:
            penalty += 0.15

        return _clip(penalty, 0.0, 0.35)


__all__ = ["EvidenceScorer"]