from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field, field_validator


class HotPotQAExample(BaseModel):
    source_question_id: str
    question: str
    answer: str
    question_type: str | None = None
    level: str | None = None
    supporting_facts: list[tuple[str, int]]
    context: list[tuple[str, list[str]]]

    @field_validator("source_question_id", "question", "answer")
    @classmethod
    def validate_non_empty_string(cls, value: str) -> str:
        if not isinstance(value, str) or not value.strip():
            raise ValueError("Field must be a non-empty string.")
        return value.strip()

    @field_validator("supporting_facts")
    @classmethod
    def validate_supporting_facts(
        cls,
        value: list[tuple[str, int]],
    ) -> list[tuple[str, int]]:
        for title, sentence_id in value:
            if not isinstance(title, str) or not title.strip():
                raise ValueError("Supporting fact title must be a non-empty string.")
            if not isinstance(sentence_id, int) or sentence_id < 0:
                raise ValueError("Supporting fact sentence index must be a non-negative integer.")
        return value

    @field_validator("context")
    @classmethod
    def validate_context(
        cls,
        value: list[tuple[str, list[str]]],
    ) -> list[tuple[str, list[str]]]:
        if not value:
            raise ValueError("Context must contain at least one document.")

        for title, sentences in value:
            if not isinstance(title, str) or not title.strip():
                raise ValueError("Context document title must be a non-empty string.")
            if not isinstance(sentences, list):
                raise ValueError("Context document sentences must be a list.")
            if not all(isinstance(sentence, str) for sentence in sentences):
                raise ValueError("Every context sentence must be a string.")

        return value


class SentenceMetadata(BaseModel):
    sentence_id: int = Field(ge=0)
    text: str
    start_char: int = Field(ge=0)
    end_char: int = Field(ge=0)

    @field_validator("text")
    @classmethod
    def validate_sentence_text(cls, value: str) -> str:
        if not isinstance(value, str):
            raise ValueError("Sentence text must be a string.")
        return value


class ParagraphChunk(BaseModel):
    chunk_id: str
    source_question_id: str
    question: str
    answer: str
    question_type: str | None = None
    level: str | None = None
    doc_title: str
    paragraph_index: int = Field(ge=0)
    chunk_text: str
    paragraph_text: str
    sentences: list[SentenceMetadata]
    supporting_sentence_ids: list[int]
    is_supporting_doc: bool

    def to_json_dict(self) -> dict[str, Any]:
        return self.model_dump()