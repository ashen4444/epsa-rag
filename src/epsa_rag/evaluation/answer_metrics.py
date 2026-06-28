from __future__ import annotations

import re
import string
from collections import Counter
from dataclasses import dataclass

_ARTICLES = re.compile(r"\b(a|an|the)\b", flags=re.IGNORECASE)


@dataclass(frozen=True)
class AnswerOverlapMetrics:
    precision: float
    recall: float
    f1: float


def normalize_answer(text: str | None) -> str:
    if text is None:
        return ""

    lowered = text.lower()
    no_punctuation = "".join(ch for ch in lowered if ch not in string.punctuation)
    no_articles = _ARTICLES.sub(" ", no_punctuation)

    return " ".join(no_articles.split())


def exact_match_score(prediction: str | None, gold_answer: str | None) -> float:
    return float(normalize_answer(prediction) == normalize_answer(gold_answer))


def partial_match_score(prediction: str | None, gold_answer: str | None) -> float:
    pred = normalize_answer(prediction)
    gold = normalize_answer(gold_answer)

    if not pred or not gold:
        return 0.0

    return float(pred in gold or gold in pred)


def answer_overlap_metrics(
    prediction: str | None,
    gold_answer: str | None,
) -> AnswerOverlapMetrics:
    pred_tokens = normalize_answer(prediction).split()
    gold_tokens = normalize_answer(gold_answer).split()

    if not pred_tokens and not gold_tokens:
        return AnswerOverlapMetrics(precision=1.0, recall=1.0, f1=1.0)

    if not pred_tokens or not gold_tokens:
        return AnswerOverlapMetrics(precision=0.0, recall=0.0, f1=0.0)

    common = Counter(pred_tokens) & Counter(gold_tokens)
    num_same = sum(common.values())

    if num_same == 0:
        return AnswerOverlapMetrics(precision=0.0, recall=0.0, f1=0.0)

    precision = num_same / len(pred_tokens)
    recall = num_same / len(gold_tokens)
    f1 = 2 * precision * recall / (precision + recall)

    return AnswerOverlapMetrics(
        precision=precision,
        recall=recall,
        f1=f1,
    )


def token_f1_score(prediction: str | None, gold_answer: str | None) -> float:
    return answer_overlap_metrics(prediction, gold_answer).f1