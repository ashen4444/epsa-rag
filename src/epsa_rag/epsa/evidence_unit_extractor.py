from __future__ import annotations

import re
from dataclasses import is_dataclass
from typing import Any, Iterable

from epsa_rag.epsa.schemas import EvidenceUnit


_PRONOUN_PATTERN = re.compile(
    r"^(?P<pronoun>he|she|it|they|his|her|its|their|him|them)\b",
    re.IGNORECASE,
)

_CLEAR_LEADING_PRONOUN_PATTERN = re.compile(
    r"^(?P<prefix>(?:in|during|after|before|later|then|also|however),?\s+)"
    r"(?P<pronoun>he|she|it|they|his|her|its|their|him|them)\b",
    re.IGNORECASE,
)

_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "in",
    "into", "is", "it", "its", "of", "on", "or", "that", "the", "their",
    "then", "there", "this", "to", "was", "were", "which", "who", "whom",
    "whose", "with", "what", "when", "where", "why", "how", "did", "does",
    "do", "had", "has", "have",
}

_RELATION_PATTERNS: dict[str, tuple[str, ...]] = {
    "born": (r"\bborn\b", r"\bbirthplace\b", r"\bborn in\b"),
    "birthplace": (r"\bbirthplace\b", r"\bplace of birth\b"),
    "directed": (r"\bdirected\b", r"\bdirector\b", r"\bdirected by\b"),
    "written": (r"\bwritten\b", r"\bwriter\b", r"\bauthor\b", r"\bnovelist\b"),
    "author": (r"\bauthor\b", r"\bwritten by\b"),
    "located": (r"\blocated\b", r"\blocated in\b", r"\bbased in\b"),
    "founded": (r"\bfounded\b", r"\bfounder\b", r"\bestablished\b"),
    "published": (r"\bpublished\b", r"\bpublisher\b"),
    "released": (r"\breleased\b", r"\brelease date\b"),
    "starring": (r"\bstarring\b", r"\bstarred\b", r"\bcast\b"),
    "member": (r"\bmember\b", r"\bmembers\b", r"\bpart of\b"),
    "capital": (r"\bcapital\b",),
    "population": (r"\bpopulation\b",),
    "genre": (r"\bgenre\b",),
    "occupation": (r"\boccupation\b", r"\bprofession\b"),
    "spouse": (r"\bspouse\b", r"\bmarried\b", r"\bwife\b", r"\bhusband\b"),
    "parent": (r"\bparent\b", r"\bfather\b", r"\bmother\b"),
    "child": (r"\bchild\b", r"\bson\b", r"\bdaughter\b"),
    "educated": (r"\beducated\b", r"\bstudied\b", r"\battended\b"),
    "alma mater": (r"\balma mater\b",),
    "discovered": (r"\bdiscovered\b", r"\bdiscovery\b"),
    "capacity": (r"\bcapacity\b", r"\bseats\b", r"\bseat\b"),
}

_ORG_SUFFIX_PATTERN = re.compile(
    r"\b(?:University|College|Institute|School|Hospital|Bank|Company|"
    r"Corporation|Corp\.?|Inc\.?|Ltd\.?|Association|Club|FC|Agency|"
    r"Ministry|Department|Committee|Council|League|Museum|Theatre|"
    r"Center|Centre)\b"
)

_DATE_PATTERN = re.compile(
    r"\b(?:\d{1,2}\s+)?(?:January|February|March|April|May|June|July|"
    r"August|September|October|November|December)\s+\d{1,2},?\s+\d{4}\b"
    r"|\b\d{4}\b"
)

_NUMBER_PATTERN = re.compile(r"\b\d+(?:,\d{3})*(?:\.\d+)?\b")
_QUOTED_PATTERN = re.compile(r"[\"“”'‘’]([^\"“”'‘’]{2,})[\"“”'‘’]")

_CAPITALIZED_PHRASE_PATTERN = re.compile(
    r"\b(?:[A-Z][\w.&'’-]*)(?:\s+(?:of|the|and|de|da|del|van|von|la|le|[A-Z][\w.&'’-]*))*"
)


def _get_value(obj: Any, key: str, default: Any = None) -> Any:
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(key, default)
    if is_dataclass(obj):
        return getattr(obj, key, default)
    return getattr(obj, key, default)


def _as_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _dedupe_preserve_order(values: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []

    for value in values:
        cleaned = _as_text(value).strip(" ,.;:()[]{}")
        if not cleaned:
            continue

        key = cleaned.casefold()
        if key not in seen:
            seen.add(key)
            result.append(cleaned)

    return result


def _tokenize(text: str) -> list[str]:
    return [
        token
        for token in re.findall(r"[a-z0-9]+", text.casefold())
        if token not in _STOPWORDS and len(token) > 1
    ]


def _extract_entities(text: str, doc_title: str) -> list[str]:
    entities: list[str] = []

    if doc_title:
        entities.append(doc_title)

    entities.extend(match.group(1) for match in _QUOTED_PATTERN.finditer(text))

    for match in _CAPITALIZED_PHRASE_PATTERN.finditer(text):
        phrase = match.group(0).strip()

        if phrase.casefold() in _STOPWORDS:
            continue

        if phrase.casefold() in {
            "he", "she", "it", "they", "his", "her", "its", "their", "him", "them"
        }:
            continue

        if len(phrase) <= 1:
            continue

        entities.append(phrase)

    return _dedupe_preserve_order(entities)


def _extract_relation_hints(text: str) -> list[str]:
    lowered = text.casefold()
    hints: list[str] = []

    for hint, patterns in _RELATION_PATTERNS.items():
        if any(re.search(pattern, lowered, flags=re.IGNORECASE) for pattern in patterns):
            hints.append(hint)

    return hints


def _extract_answer_type_candidates(
    text: str,
    entities: list[str],
    relation_hints: list[str],
) -> list[str]:
    candidates: list[str] = []

    if _DATE_PATTERN.search(text):
        candidates.append("DATE")

    if _NUMBER_PATTERN.search(text):
        candidates.append("NUMBER")

    if {"born", "birthplace", "located", "capital"}.intersection(relation_hints):
        candidates.append("LOCATION")

    if any(_ORG_SUFFIX_PATTERN.search(entity) for entity in entities):
        candidates.append("ORGANIZATION")

    person_like_entities = [entity for entity in entities if len(entity.split()) >= 2]
    if person_like_entities and not any(
        _ORG_SUFFIX_PATTERN.search(entity) for entity in person_like_entities
    ):
        candidates.append("PERSON")

    if _QUOTED_PATTERN.search(text) or {
        "released", "published", "written", "directed", "starring", "genre"
    }.intersection(relation_hints):
        candidates.append("TITLE_OR_WORK")

    if entities:
        candidates.append("ENTITY")

    return _dedupe_preserve_order(candidates)


def _resolve_leading_pronoun(sentence_text: str, doc_title: str) -> str:
    text = sentence_text.strip()

    if not text or not doc_title:
        return text

    direct_match = _PRONOUN_PATTERN.match(text)
    if direct_match:
        pronoun = direct_match.group("pronoun")
        replacement = doc_title

        if pronoun.casefold() in {"his", "her", "its", "their"}:
            replacement = f"{doc_title}'s"

        return f"{replacement}{text[direct_match.end():]}".strip()

    leading_match = _CLEAR_LEADING_PRONOUN_PATTERN.match(text)
    if leading_match:
        pronoun = leading_match.group("pronoun")
        replacement = doc_title

        if pronoun.casefold() in {"his", "her", "its", "their"}:
            replacement = f"{doc_title}'s"

        start, end = leading_match.span("pronoun")
        return f"{text[:start]}{replacement}{text[end:]}".strip()

    return text


def _normalize_sentences(chunk: Any) -> list[dict[str, Any]]:
    raw_sentences = _get_value(chunk, "sentences", None)

    if raw_sentences:
        normalized: list[dict[str, Any]] = []

        for index, sentence in enumerate(raw_sentences):
            if isinstance(sentence, dict):
                text = _as_text(sentence.get("text") or sentence.get("sentence_text"))
                sentence_id = int(sentence.get("sentence_id", index))
                is_supporting = sentence.get("is_supporting_sentence")
            else:
                text = _as_text(sentence)
                sentence_id = index
                is_supporting = None

            if text:
                normalized.append(
                    {
                        "sentence_id": sentence_id,
                        "text": text,
                        "is_supporting_sentence": is_supporting,
                    }
                )

        if normalized:
            return normalized

    fallback_text = (
        _as_text(_get_value(chunk, "paragraph_text"))
        or _as_text(_get_value(chunk, "chunk_text"))
        or _as_text(_get_value(chunk, "text"))
    )

    if fallback_text:
        return [{"sentence_id": 0, "text": fallback_text, "is_supporting_sentence": None}]

    return []


class EvidenceUnitExtractor:
    """
    Converts retrieved paragraph chunks into deterministic sentence-level EPSA evidence units.

    This module does not retrieve, score sufficiency, build graphs, prune context,
    or call an LLM.
    """

    def extract_from_chunk(
        self,
        candidate_evidence: Any,
        chunk: Any,
        question_analysis: Any,
    ) -> list[EvidenceUnit]:
        chunk_id = _as_text(
            _get_value(candidate_evidence, "chunk_id") or _get_value(chunk, "chunk_id")
        )
        doc_title = _as_text(
            _get_value(candidate_evidence, "doc_title") or _get_value(chunk, "doc_title")
        )
        paragraph_index = int(
            _get_value(
                candidate_evidence,
                "paragraph_index",
                _get_value(chunk, "paragraph_index", 0),
            )
            or 0
        )
        retrieval_rank = _get_value(candidate_evidence, "retrieval_rank", None)
        retrieval_score = _get_value(candidate_evidence, "retrieval_score", None)

        supporting_ids = set(_get_value(chunk, "supporting_sentence_ids", []) or [])

        question_entities = [
            _as_text(entity)
            for entity in (_get_value(question_analysis, "seed_entities", []) or [])
        ]

        question_text = _as_text(
            _get_value(question_analysis, "normalized_question")
            or _get_value(question_analysis, "question")
            or " ".join(question_entities)
        )

        question_tokens = set(_tokenize(question_text))

        evidence_units: list[EvidenceUnit] = []

        for sentence in _normalize_sentences(chunk):
            sentence_id = int(sentence["sentence_id"])
            sentence_text = sentence["text"]
            resolved_text = _resolve_leading_pronoun(sentence_text, doc_title)

            entities = _extract_entities(resolved_text, doc_title)
            relation_hints = _extract_relation_hints(resolved_text)
            answer_type_candidates = _extract_answer_type_candidates(
                resolved_text,
                entities,
                relation_hints,
            )

            sentence_tokens = set(_tokenize(resolved_text))

            question_entity_overlap = [
                entity
                for entity in question_entities
                if entity and entity.casefold() in resolved_text.casefold()
            ]

            token_overlap = 0.0
            if question_tokens:
                token_overlap = len(question_tokens.intersection(sentence_tokens)) / len(question_tokens)

            explicit_supporting = sentence.get("is_supporting_sentence")
            if explicit_supporting is None:
                is_supporting_sentence = sentence_id in supporting_ids if supporting_ids else None
            else:
                is_supporting_sentence = bool(explicit_supporting)

            evidence_units.append(
                EvidenceUnit(
                    evidence_unit_id=f"{chunk_id}::s{sentence_id}",
                    chunk_id=chunk_id,
                    doc_title=doc_title,
                    paragraph_index=paragraph_index,
                    sentence_id=sentence_id,
                    sentence_text=sentence_text,
                    resolved_text=resolved_text,
                    entities=entities,
                    relation_hints=relation_hints,
                    answer_type_candidates=answer_type_candidates,
                    question_entity_overlap=_dedupe_preserve_order(question_entity_overlap),
                    question_token_overlap=round(token_overlap, 6),
                    is_supporting_sentence=is_supporting_sentence,
                    retrieval_rank=int(retrieval_rank) if retrieval_rank is not None else None,
                    retrieval_score=float(retrieval_score) if retrieval_score is not None else None,
                )
            )

        return evidence_units

    def extract_many(
        self,
        candidate_chunk_pairs: Iterable[tuple[Any, Any]],
        question_analysis: Any,
    ) -> list[EvidenceUnit]:
        units: list[EvidenceUnit] = []

        for candidate_evidence, chunk in candidate_chunk_pairs:
            units.extend(
                self.extract_from_chunk(
                    candidate_evidence=candidate_evidence,
                    chunk=chunk,
                    question_analysis=question_analysis,
                )
            )

        return units


__all__ = ["EvidenceUnitExtractor"]