"""Rule-based question analysis for EPSA.

The analyzer intentionally avoids LLM calls. It extracts transparent features
that downstream EPSA modules can use for evidence scoring, graph construction,
sufficiency checks, and next-hop query generation.
"""

from __future__ import annotations

import re
from collections.abc import Iterable

from epsa_rag.epsa.schemas import (
    AnswerType,
    AnswerTypeCandidate,
    EntityMention,
    QuestionAnalysis,
    QuestionType,
    RelationHint,
)


QUESTION_LEAD_WORDS = {
    "what",
    "which",
    "who",
    "whom",
    "whose",
    "where",
    "when",
    "why",
    "how",
    "is",
    "are",
    "was",
    "were",
    "do",
    "does",
    "did",
    "can",
    "could",
    "has",
    "have",
    "had",
}

YES_NO_STARTERS = (
    "is ",
    "are ",
    "was ",
    "were ",
    "do ",
    "does ",
    "did ",
    "can ",
    "could ",
    "has ",
    "have ",
    "had ",
)

TITLE_OR_WORK_TERMS = (
    "film",
    "movie",
    "book",
    "novel",
    "album",
    "song",
    "magazine",
    "series",
    "television series",
    "tv series",
    "play",
    "poem",
)

RELATION_PATTERNS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("born", (r"\bborn\b", r"\bbirthplace\b", r"\bborn in\b")),
    ("directed", (r"\bdirected\b", r"\bdirector\b")),
    ("written", (r"\bwritten\b", r"\bwriter\b", r"\bauthor\b", r"\bnovelist\b")),
    ("located", (r"\blocated\b", r"\bbased in\b", r"\bin\b", r"\bat\b")),
    ("founded", (r"\bfounded\b", r"\bfounder\b", r"\bestablished\b", r"\bstarted\b")),
    ("published", (r"\bpublished\b", r"\bpublisher\b")),
    ("released", (r"\breleased\b", r"\brelease date\b")),
    ("starring", (r"\bstarring\b", r"\bstarred\b", r"\bcast\b")),
    ("member", (r"\bmember\b", r"\bpart of\b", r"\bbelongs to\b")),
    ("capital", (r"\bcapital\b",)),
    ("population", (r"\bpopulation\b",)),
    ("genre", (r"\bgenre\b",)),
    ("occupation", (r"\boccupation\b", r"\bprofession\b")),
    ("spouse", (r"\bspouse\b", r"\bmarried\b", r"\bwife\b", r"\bhusband\b")),
    ("parent", (r"\bparent\b", r"\bfather\b", r"\bmother\b")),
    ("child", (r"\bchild\b", r"\bson\b", r"\bdaughter\b")),
    ("educated", (r"\beducated\b", r"\balma mater\b", r"\battended\b")),
)


class QuestionAnalyzer:
    """Deterministic EPSA question analyzer."""

    def analyze(self, question: str) -> QuestionAnalysis:
        """Analyze a natural-language question into EPSA-ready features."""
        if not isinstance(question, str) or not question.strip():
            raise ValueError("question must be a non-empty string")

        raw_question = question.strip()
        normalized_question = normalize_text(raw_question)
        expected_answer_type = infer_expected_answer_type(normalized_question)
        question_type = infer_question_type(normalized_question, expected_answer_type)
        seed_entities = extract_entity_mentions(raw_question, source="question")
        relation_hints = extract_relation_hints(raw_question, source="question")
        comparison_targets = extract_comparison_targets(
            raw_question=raw_question,
            normalized_question=normalized_question,
            seed_entities=seed_entities,
            question_type=question_type,
        )
        answer_type_candidates = [
            AnswerTypeCandidate(
                answer_type=expected_answer_type,
                text=expected_answer_type.value,
                source="question_expected_type",
                confidence=0.75 if expected_answer_type != AnswerType.UNKNOWN else 0.35,
            )
        ]

        return QuestionAnalysis(
            raw_question=raw_question,
            normalized_question=normalized_question,
            question_type=question_type,
            expected_answer_type=expected_answer_type,
            seed_entities=seed_entities,
            required_relation_hints=relation_hints,
            comparison_targets=comparison_targets,
            answer_type_candidates=answer_type_candidates,
            metadata={"analyzer": self.__class__.__name__, "version": "rule_based_v1"},
        )



def normalize_text(text: str) -> str:
    """Normalize text for stable rule matching."""
    text = text.strip().replace("“", '"').replace("”", '"').replace("’", "'")
    text = re.sub(r"\s+", " ", text)
    return text.lower()



def normalize_entity(text: str) -> str:
    """Normalize an entity-like string for matching."""
    text = normalize_text(text)
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()



def infer_expected_answer_type(normalized_question: str) -> AnswerType:
    """Infer the expected answer type from the question wording."""
    if normalized_question.startswith(YES_NO_STARTERS):
        return AnswerType.BOOLEAN
    if re.search(r"\b(where|what place|which city|which country)\b", normalized_question):
        return AnswerType.LOCATION
    if re.search(r"\b(when|what year|which year|what date|which date)\b", normalized_question):
        return AnswerType.DATE
    if re.search(r"\b(how many|how much|number of|population|age|height|length|distance)\b", normalized_question):
        return AnswerType.NUMBER
    if re.search(r"\b(who|whom|whose)\b", normalized_question):
        return AnswerType.PERSON
    if any(re.search(rf"\bwhich {re.escape(term)}\b", normalized_question) for term in TITLE_OR_WORK_TERMS):
        return AnswerType.TITLE_OR_WORK
    if re.search(r"\b(which company|which organization|which university|which team|which band)\b", normalized_question):
        return AnswerType.ORGANIZATION
    if normalized_question.startswith("which ") or normalized_question.startswith("what "):
        return AnswerType.ENTITY
    return AnswerType.UNKNOWN



def infer_question_type(normalized_question: str, expected_answer_type: AnswerType) -> QuestionType:
    """Infer EPSA question type using transparent lexical rules."""
    comparison_markers = (
        " both ",
        "which one",
        "which of",
        "same ",
        "same type",
        "older",
        "younger",
        "larger",
        "smaller",
        "earlier",
        "later",
        "first",
        "started first",
        "born first",
        "founded first",
        "released first",
        "more ",
        "less ",
        "higher",
        "lower",
        "between ",
    )
    padded = f" {normalized_question} "
    if any(marker in padded for marker in comparison_markers):
        return QuestionType.COMPARISON

    # HotPotQA often phrases binary comparison as:
    #   "Which X ... A or B?"
    #   "Who ... David Lee Roth or Cia Berg?"
    # Treat these as comparison/choice questions instead of single-hop factoids.
    if (
        re.search(r"\b(which|who|what)\b", normalized_question)
        and " or " in padded
        and normalized_question.endswith("?")
    ):
        return QuestionType.COMPARISON

    if expected_answer_type == AnswerType.BOOLEAN:
        return QuestionType.YES_NO

    bridge_patterns = (
        r"\b(where|when|who|what|which)\b.+\b(the|a|an)\b.+\bof\b.+",
        r"\bof the\b",
        r"\bwhose\b.+\b(was|is|were|are)\b",
        r"\b(person|actor|director|author|writer|founder|capital|city|country)\b.+\bof\b",
    )
    if any(re.search(pattern, normalized_question) for pattern in bridge_patterns):
        return QuestionType.BRIDGE

    return QuestionType.FACTOID



def extract_entity_mentions(text: str, source: str) -> list[EntityMention]:
    """Extract quoted strings and capitalized/title-like spans."""
    mentions: list[EntityMention] = []
    seen: set[str] = set()

    quoted_patterns = (r"\"([^\"]+)\"", r"'([^']+)'")
    for pattern in quoted_patterns:
        for match in re.finditer(pattern, text):
            entity_text = match.group(1).strip()
            _add_entity(mentions, seen, entity_text, source, match.start(1), match.end(1), 0.98)

    # Multi-token capitalized phrases, allowing small connector words inside titles.
    word = r"(?:[A-Z][A-Za-z0-9'&-]*(?:\.[A-Z][A-Za-z0-9'&-]*)*|[A-Z]{2,})"
    connector = r"(?:of|the|and|for|in|on|at|de|la|le|du|&)"
    multi_pattern = re.compile(rf"\b{word}(?:[ \t]+(?:{connector}|{word}))+")
    for match in multi_pattern.finditer(text):
        entity_text = _clean_entity_text(match.group(0))
        if not _is_question_lead(entity_text):
            _add_entity(mentions, seen, entity_text, source, match.start(), match.end(), 0.9)

    # Single title-like tokens are useful for HotPotQA titles such as Inception.
    single_pattern = re.compile(r"\b[A-Z][A-Za-z0-9'&.-]{2,}\b")
    for match in single_pattern.finditer(text):
        entity_text = _clean_entity_text(match.group(0))
        if not _is_question_lead(entity_text):
            _add_entity(mentions, seen, entity_text, source, match.start(), match.end(), 0.7)

    return mentions



def extract_relation_hints(text: str, source: str) -> list[RelationHint]:
    """Extract transparent relation hints from text."""
    hints: list[RelationHint] = []
    normalized = normalize_text(text)
    seen: set[tuple[str, str, int]] = set()

    for relation, patterns in RELATION_PATTERNS:
        for pattern in patterns:
            for match in re.finditer(pattern, normalized, flags=re.IGNORECASE):
                key = (relation, match.group(0), match.start())
                if key in seen:
                    continue
                seen.add(key)
                hints.append(
                    RelationHint(
                        relation=relation,
                        matched_text=match.group(0),
                        source=source,
                        start_char=match.start(),
                        end_char=match.end(),
                        confidence=0.8,
                    )
                )
    return hints



def extract_comparison_targets(
    raw_question: str,
    normalized_question: str,
    seed_entities: Iterable[EntityMention],
    question_type: QuestionType,
) -> list[EntityMention]:
    """Extract entities being compared when the question is comparative."""
    if question_type != QuestionType.COMPARISON:
        return []

    seeds = list(seed_entities)
    which_of_match = re.search(
        r"which\s+of\s+(.+?)\s+(?:was|were|is|are|has|have|had|did|does|do|released|came|comes)\b",
        raw_question,
        re.IGNORECASE,
    )
    if which_of_match and " and " in which_of_match.group(1):
        left, right = which_of_match.group(1).rsplit(" and ", maxsplit=1)
        targets = []
        for text in (left, right):
            text = _clean_entity_text(text)
            if text:
                targets.append(
                    EntityMention(
                        text=text,
                        normalized=normalize_entity(text),
                        source="question_comparison_target",
                        confidence=0.8,
                    )
                )
        return targets

    if len(seeds) >= 2:
        return seeds[:2]

    between_match = re.search(r"between\s+(.+?)\s+and\s+(.+?)(?:\?|$)", raw_question, re.IGNORECASE)
    if between_match:
        targets: list[EntityMention] = []
        for group_idx in (1, 2):
            text = _clean_entity_text(between_match.group(group_idx))
            if text:
                targets.append(
                    EntityMention(
                        text=text,
                        normalized=normalize_entity(text),
                        source="question_comparison_target",
                        confidence=0.65,
                    )
                )
        return targets

    # Fallback: keep the evidence honest instead of overclaiming.
    return seeds



def _add_entity(
    mentions: list[EntityMention],
    seen: set[str],
    text: str,
    source: str,
    start_char: int | None,
    end_char: int | None,
    confidence: float,
) -> None:
    text = _clean_entity_text(text)
    normalized = normalize_entity(text)
    if not text or not normalized or normalized in seen:
        return
    seen.add(normalized)
    mentions.append(
        EntityMention(
            text=text,
            normalized=normalized,
            source=source,
            start_char=start_char,
            end_char=end_char,
            confidence=confidence,
        )
    )



def _clean_entity_text(text: str) -> str:
    text = re.sub(r"\s+", " ", text).strip(" \t\n\r,.;:!?()[]{}")
    return text



def _is_question_lead(text: str) -> bool:
    return normalize_entity(text).split(" ", maxsplit=1)[0] in QUESTION_LEAD_WORDS
