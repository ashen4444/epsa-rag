"""Rule-based candidate chunk evidence analysis for EPSA."""

from __future__ import annotations

import re
from dataclasses import is_dataclass
from typing import Any

from epsa_rag.epsa.question_analyzer import (
    extract_entity_mentions,
    extract_relation_hints,
    normalize_entity,
)
from epsa_rag.epsa.schemas import (
    AnswerType,
    AnswerTypeCandidate,
    CandidateChunkEvidence,
    EntityMention,
    QuestionAnalysis,
)


TOKEN_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "did",
    "do",
    "does",
    "for",
    "from",
    "had",
    "has",
    "have",
    "how",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "the",
    "their",
    "this",
    "to",
    "was",
    "were",
    "what",
    "when",
    "where",
    "which",
    "who",
    "whose",
    "with",
}


class CandidateChunkEvidenceAnalyzer:
    """Extract EPSA-ready evidence features from a retrieved paragraph chunk."""

    def analyze(
        self,
        chunk: Any,
        question_analysis: QuestionAnalysis | None = None,
        retrieval_rank: int | None = None,
        retrieval_score: float | None = None,
    ) -> CandidateChunkEvidence:
        """Analyze one paragraph chunk.

        The function accepts dict-like chunks, dataclass/object-like chunks, or
        retrieval-result-like objects containing the standard corpus fields.
        """
        chunk_id = str(_first_present(chunk, "chunk_id", "id") or "")
        doc_title = str(_first_present(chunk, "doc_title", "title") or "")
        paragraph_index = _first_present(chunk, "paragraph_index", "paragraph_id")
        source_question_id = _first_present(chunk, "question_id", "source_question_id")
        sentences = _first_present(chunk, "sentences") or []

        object_rank = _first_present(chunk, "rank", "retrieval_rank")
        object_score = _first_present(chunk, "score", "retrieval_score", "fusion_score")
        retrieval_rank = retrieval_rank if retrieval_rank is not None else _safe_int(object_rank)
        retrieval_score = retrieval_score if retrieval_score is not None else _safe_float(object_score)

        paragraph_text = str(_first_present(chunk, "paragraph_text") or "")
        chunk_text = str(_first_present(chunk, "chunk_text", "text") or "")
        if not paragraph_text and chunk_text:
            paragraph_text = _strip_title_prefix(chunk_text)
        if not chunk_text:
            chunk_text = _compose_chunk_text(doc_title=doc_title, paragraph_text=paragraph_text)

        body_text = paragraph_text or _strip_title_prefix(chunk_text)
        analysis_text = "\n".join(part for part in (doc_title, body_text or chunk_text) if part)

        entities = self._extract_entities(doc_title=doc_title, text=body_text or chunk_text)
        relation_hints = extract_relation_hints(analysis_text, source="chunk")
        answer_type_candidates = self._extract_answer_type_candidates(analysis_text)
        question_entity_overlap = self._compute_question_entity_overlap(entities, question_analysis)
        question_token_overlap, token_overlap_score = self._compute_question_token_overlap(
            analysis_text=analysis_text,
            question_analysis=question_analysis,
        )
        is_title_match = self._is_title_match(doc_title, question_analysis, question_entity_overlap)
        potential_bridge_entities = self._extract_potential_bridge_entities(
            entities=entities,
            question_analysis=question_analysis,
            doc_title=doc_title,
        )

        return CandidateChunkEvidence(
            chunk_id=chunk_id,
            doc_title=doc_title,
            paragraph_index=_safe_int(paragraph_index),
            retrieval_rank=retrieval_rank,
            retrieval_score=retrieval_score,
            entities=entities,
            relation_hints=relation_hints,
            answer_type_candidates=answer_type_candidates,
            potential_bridge_entities=potential_bridge_entities,
            question_entity_overlap=question_entity_overlap,
            question_token_overlap=question_token_overlap,
            question_token_overlap_score=token_overlap_score,
            is_title_match=is_title_match,
            chunk_text=chunk_text,
            paragraph_text=paragraph_text,
            source_question_id=str(source_question_id) if source_question_id is not None else None,
            sentences=list(sentences) if isinstance(sentences, (list, tuple)) else [],
            metadata={"analyzer": self.__class__.__name__, "version": "rule_based_v1"},
        )

    def _extract_entities(self, doc_title: str, text: str) -> list[EntityMention]:
        entities: list[EntityMention] = []
        seen: set[str] = set()

        if doc_title.strip():
            title_norm = normalize_entity(doc_title)
            if title_norm:
                entities.append(
                    EntityMention(
                        text=doc_title.strip(),
                        normalized=title_norm,
                        source="doc_title",
                        confidence=1.0,
                    )
                )
                seen.add(title_norm)

        for mention in extract_entity_mentions(text, source="chunk"):
            if mention.normalized in seen:
                continue
            seen.add(mention.normalized)
            entities.append(mention)
        return entities

    def _extract_answer_type_candidates(self, text: str) -> list[AnswerTypeCandidate]:
        candidates: list[AnswerTypeCandidate] = []
        seen: set[tuple[AnswerType, str]] = set()

        patterns: list[tuple[AnswerType, str, float]] = [
            (AnswerType.DATE, r"\b(?:1[5-9]\d{2}|20\d{2}|21\d{2})\b", 0.9),
            (AnswerType.DATE, r"\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},\s+\d{4}\b", 0.95),
            (AnswerType.NUMBER, r"\b\d+(?:,\d{3})*(?:\.\d+)?\b", 0.75),
            (AnswerType.LOCATION, r"\b(?:in|at|from|located in|based in)\s+([A-Z][A-Za-z'&.-]+(?:\s+[A-Z][A-Za-z'&.-]+){0,4})", 0.65),
        ]
        for answer_type, pattern, confidence in patterns:
            for match in re.finditer(pattern, text):
                candidate_text = match.group(1) if match.lastindex else match.group(0)
                candidate_text = candidate_text.strip(" ,.;:!?()[]{}")
                key = (answer_type, candidate_text.lower())
                if not candidate_text or key in seen:
                    continue
                seen.add(key)
                candidates.append(
                    AnswerTypeCandidate(
                        answer_type=answer_type,
                        text=candidate_text,
                        source="chunk_pattern",
                        start_char=match.start(1) if match.lastindex else match.start(),
                        end_char=match.end(1) if match.lastindex else match.end(),
                        confidence=confidence,
                    )
                )

        # Entity-like spans can become PERSON/ENTITY candidates downstream.
        for mention in extract_entity_mentions(text, source="chunk_candidate"):
            if mention.source == "doc_title":
                continue
            key = (AnswerType.ENTITY, mention.normalized)
            if key in seen:
                continue
            seen.add(key)
            candidates.append(
                AnswerTypeCandidate(
                    answer_type=AnswerType.ENTITY,
                    text=mention.text,
                    source="chunk_entity_like_span",
                    start_char=mention.start_char,
                    end_char=mention.end_char,
                    confidence=0.55,
                )
            )
        return candidates

    def _compute_question_entity_overlap(
        self,
        entities: list[EntityMention],
        question_analysis: QuestionAnalysis | None,
    ) -> list[str]:
        if question_analysis is None:
            return []
        question_entities = list(question_analysis.seed_entities) + list(question_analysis.comparison_targets)
        overlap: list[str] = []
        seen: set[str] = set()
        for chunk_entity in entities:
            for question_entity in question_entities:
                if _entity_matches(chunk_entity.normalized, question_entity.normalized):
                    if chunk_entity.normalized not in seen:
                        overlap.append(chunk_entity.text)
                        seen.add(chunk_entity.normalized)
        return overlap

    def _compute_question_token_overlap(
        self,
        analysis_text: str,
        question_analysis: QuestionAnalysis | None,
    ) -> tuple[list[str], float]:
        if question_analysis is None:
            return [], 0.0
        question_tokens = _content_tokens(question_analysis.normalized_question)
        text_tokens = set(_content_tokens(analysis_text.lower()))
        overlap = [token for token in question_tokens if token in text_tokens]
        # Add transparent lexical equivalents for common relation wording differences.
        equivalents = {
            "director": {"directed", "director"},
            "writer": {"written", "writer", "author"},
            "author": {"written", "writer", "author"},
            "birthplace": {"born", "birthplace"},
        }
        lowered_text = analysis_text.lower()
        for question_token, text_variants in equivalents.items():
            if question_token in question_tokens and any(variant in lowered_text for variant in text_variants):
                overlap.append(question_token)

        unique_overlap = list(dict.fromkeys(overlap))
        score = len(set(unique_overlap)) / max(len(set(question_tokens)), 1)
        return unique_overlap, round(score, 4)

    def _is_title_match(
        self,
        doc_title: str,
        question_analysis: QuestionAnalysis | None,
        question_entity_overlap: list[str],
    ) -> bool:
        if not doc_title or question_analysis is None:
            return False
        title_norm = normalize_entity(doc_title)
        if not title_norm:
            return False
        if title_norm in {normalize_entity(item) for item in question_entity_overlap}:
            return True
        if title_norm and title_norm in normalize_entity(question_analysis.raw_question):
            return True
        for entity in question_analysis.seed_entities + question_analysis.comparison_targets:
            if _entity_matches(title_norm, entity.normalized):
                return True
        return False

    def _extract_potential_bridge_entities(
        self,
        entities: list[EntityMention],
        question_analysis: QuestionAnalysis | None,
        doc_title: str,
    ) -> list[EntityMention]:
        title_norm = normalize_entity(doc_title)
        question_norms = set()
        if question_analysis is not None:
            question_norms = {
                entity.normalized
                for entity in question_analysis.seed_entities + question_analysis.comparison_targets
            }

        bridges: list[EntityMention] = []
        for entity in entities:
            if entity.normalized == title_norm:
                continue
            if entity.normalized in question_norms:
                continue
            if len(entity.normalized) < 3:
                continue
            bridges.append(entity)
        return bridges



def _first_present(obj: Any, *names: str) -> Any:
    """Return the first available field/attribute from dict-like or object-like input."""
    # Some retriever wrappers store the corpus chunk under a nested attribute.
    for nested_name in ("chunk", "document", "metadata"):
        nested = _get_value(obj, nested_name)
        if nested is not None:
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



def _strip_title_prefix(chunk_text: str) -> str:
    return re.sub(r"^\s*Title:\s*.*?\n\s*Paragraph:\s*", "", chunk_text, flags=re.IGNORECASE | re.DOTALL).strip()



def _compose_chunk_text(doc_title: str, paragraph_text: str) -> str:
    if doc_title and paragraph_text:
        return f"Title: {doc_title}\nParagraph: {paragraph_text}"
    return paragraph_text or doc_title



def _safe_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None



def _safe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None



def _entity_matches(left: str, right: str) -> bool:
    if not left or not right:
        return False
    return left == right or left in right or right in left



def _content_tokens(text: str) -> list[str]:
    tokens = re.findall(r"[a-z0-9]+", text.lower())
    return [token for token in tokens if token not in TOKEN_STOPWORDS and len(token) > 1]
