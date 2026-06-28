from __future__ import annotations

from epsa_rag.rag.llm_client import ChatMessage


HOP2_QUERY_SYSTEM_PROMPT = """You are a retrieval query generator for multi-hop question answering.
Generate exactly one concise search query for the missing second-hop evidence.
Use only the question and the provided Hop-1 evidence.
Return only the search query. Do not explain."""


FINAL_ANSWER_SYSTEM_PROMPT = """You are a strict answer extraction model for HotPotQA.

Answer using only the provided retrieved context.

Rules:
1. Return only the final answer span.
2. Do not write a full sentence.
3. Do not explain.
4. Do not include dates, descriptions, or supporting details unless they are the answer.
5. For yes/no questions, return only yes or no.
6. For comparison questions, return only the entity, title, date, number, place, or value that answers the question.
7. If the context contains the answer, do not return "Insufficient evidence".
8. If the context is truly not enough to answer, return exactly: Insufficient evidence.

Examples:
Question: Are both people American?
Answer: yes

Question: Which magazine was started first?
Answer: Arthur's Magazine

Question: What year was the person born?
Answer: 1988

Question: In which game was Malcolm Smith named Most Valuable Player?
Answer: Super Bowl XLVIII
"""


def build_hop2_query_messages(question: str, hop1_context: str) -> list[ChatMessage]:
    user_prompt = f"""Question:
{question}

Hop-1 retrieved context:
{hop1_context}

Generate the best second-hop retrieval query."""
    return [
        ChatMessage(role="system", content=HOP2_QUERY_SYSTEM_PROMPT),
        ChatMessage(role="user", content=user_prompt),
    ]


def build_final_answer_messages(question: str, merged_context: str) -> list[ChatMessage]:
    user_prompt = f"""Question:
{question}

Retrieved context:
{merged_context}

Return only the final answer span."""
    return [
        ChatMessage(role="system", content=FINAL_ANSWER_SYSTEM_PROMPT),
        ChatMessage(role="user", content=user_prompt),
    ]