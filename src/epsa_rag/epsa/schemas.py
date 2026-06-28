"""Core data schemas for EPSA.

EPSA uses lightweight structured objects so later modules can share the same
representation for question analysis, candidate evidence analysis, evidence
unit extraction, scoring, graph construction, and context pruning.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Optional
from dataclasses import dataclass, field

class QuestionType(str, Enum):
    """Supported question categories used by EPSA."""

    BRIDGE = "bridge"
    COMPARISON = "comparison"
    YES_NO = "yes_no"
    FACTOID = "factoid"


class AnswerType(str, Enum):
    """Coarse answer type labels used by rule-based EPSA modules."""

    PERSON = "PERSON"
    LOCATION = "LOCATION"
    DATE = "DATE"
    NUMBER = "NUMBER"
    BOOLEAN = "BOOLEAN"
    TITLE_OR_WORK = "TITLE_OR_WORK"
    ORGANIZATION = "ORGANIZATION"
    ENTITY = "ENTITY"
    UNKNOWN = "UNKNOWN"


@dataclass(frozen=True)
class EntityMention:
    """A deterministic entity-like mention extracted from a question or chunk."""

    text: str
    normalized: str
    source: str
    start_char: Optional[int] = None
    end_char: Optional[int] = None
    confidence: float = 1.0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RelationHint:
    """A transparent relation keyword/pattern detected in text."""

    relation: str
    matched_text: str
    source: str
    start_char: Optional[int] = None
    end_char: Optional[int] = None
    confidence: float = 1.0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class AnswerTypeCandidate:
    """A candidate text span that may satisfy a coarse answer type."""

    answer_type: AnswerType
    text: str
    source: str
    start_char: Optional[int] = None
    end_char: Optional[int] = None
    confidence: float = 1.0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class QuestionAnalysis:
    """Structured output of the EPSA Question Analyzer."""

    raw_question: str
    normalized_question: str
    question_type: QuestionType
    expected_answer_type: AnswerType
    seed_entities: list[EntityMention] = field(default_factory=list)
    required_relation_hints: list[RelationHint] = field(default_factory=list)
    comparison_targets: list[EntityMention] = field(default_factory=list)
    answer_type_candidates: list[AnswerTypeCandidate] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""
        data = asdict(self)
        data["question_type"] = self.question_type.value
        data["expected_answer_type"] = self.expected_answer_type.value
        for candidate in data["answer_type_candidates"]:
            candidate["answer_type"] = candidate["answer_type"].value
        return data


@dataclass(frozen=True)
class CandidateChunkEvidence:
    """Structured evidence features extracted from one retrieved paragraph chunk."""

    chunk_id: str
    doc_title: str
    paragraph_index: Optional[int]
    retrieval_rank: Optional[int]
    retrieval_score: Optional[float]
    entities: list[EntityMention] = field(default_factory=list)
    relation_hints: list[RelationHint] = field(default_factory=list)
    answer_type_candidates: list[AnswerTypeCandidate] = field(default_factory=list)
    potential_bridge_entities: list[EntityMention] = field(default_factory=list)
    question_entity_overlap: list[str] = field(default_factory=list)
    question_token_overlap: list[str] = field(default_factory=list)
    question_token_overlap_score: float = 0.0
    is_title_match: bool = False
    chunk_text: str = ""
    paragraph_text: str = ""
    source_question_id: Optional[str] = None
    sentences: list[Any] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""
        data = asdict(self)
        for candidate in data["answer_type_candidates"]:
            candidate["answer_type"] = candidate["answer_type"].value
        return data


@dataclass(frozen=True)
class EvidenceUnit:
    evidence_unit_id: str
    chunk_id: str
    doc_title: str
    paragraph_index: int
    sentence_id: int
    sentence_text: str
    resolved_text: str
    entities: list[str] = field(default_factory=list)
    relation_hints: list[str] = field(default_factory=list)
    answer_type_candidates: list[str] = field(default_factory=list)
    question_entity_overlap: list[str] = field(default_factory=list)
    question_token_overlap: float = 0.0
    is_supporting_sentence: bool | None = None
    retrieval_rank: int | None = None
    retrieval_score: float | None = None


@dataclass(frozen=True)
class ScoredEvidenceUnit:
    evidence_unit: EvidenceUnit
    final_score: float
    score_breakdown: dict[str, float] = field(default_factory=dict)

@dataclass(frozen=True)
class EvidenceGraphNode:
    """A stable, serializable node in the EPSA evidence graph."""

    node_id: str
    node_type: str
    label: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class EvidenceGraphEdge:
    """A stable, serializable weighted edge in the EPSA evidence graph."""

    edge_id: str
    source_id: str
    target_id: str
    edge_type: str
    weight: float = 1.0
    evidence_unit_id: str | None = None
    relation: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class EvidenceGraph:
    """Deterministic graph built from scored sentence-level evidence units."""

    nodes: dict[str, EvidenceGraphNode]
    edges: list[EvidenceGraphEdge]
    question_type: str
    seed_entity_node_ids: list[str] = field(default_factory=list)
    evidence_unit_node_ids: list[str] = field(default_factory=list)
    entity_node_ids: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class EvidencePath:
    """Candidate reasoning path found in an EvidenceGraph.

    This is intentionally not a sufficiency decision. It only records a ranked
    candidate path and its provenance for the later Sufficiency Decision Engine.
    """

    path_id: str
    question_type: str
    node_ids: list[str]
    edge_ids: list[str]
    evidence_unit_ids: list[str]
    entity_chain: list[str]
    relation_chain: list[str]
    answer_candidate: str | None
    answer_type: str | None
    score: float
    metadata: dict[str, Any] = field(default_factory=dict)
