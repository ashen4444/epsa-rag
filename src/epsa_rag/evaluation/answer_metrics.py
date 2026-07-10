from __future__ import annotations

import re
import string
from collections import Counter
from dataclasses import dataclass

_ARTICLES = re.compile(r"\b(a|an|the)\b", flags=re.IGNORECASE)

_HONORIFIC_PREFIXES = {
    "doctor",
    "dr",
    "king",
    "lady",
    "lord",
    "mr",
    "mrs",
    "ms",
    "president",
    "prince",
    "princess",
    "professor",
    "queen",
    "saint",
    "sir",
    "st",
}

_MONTH_NAMES = {
    "january",
    "february",
    "march",
    "april",
    "may",
    "june",
    "july",
    "august",
    "september",
    "october",
    "november",
    "december",
}

_QUANTITY_UNITS = {
    "day",
    "days",
    "hour",
    "hours",
    "minute",
    "minutes",
    "month",
    "months",
    "second",
    "seconds",
    "week",
    "weeks",
    "year",
    "years",
}

_SAFE_PROFESSION_MODIFIERS = {
    "film",
    "movie",
    "stage",
    "television",
    "theater",
    "theatre",
    "tv",
}

_UNCERTAIN_OR_CONTRADICTORY_TOKENS = {
    "cannot",
    "different",
    "either",
    "except",
    "incorrect",
    "instead",
    "maybe",
    "never",
    "not",
    "or",
    "perhaps",
    "possibly",
    "rather",
    "unrelated",
    "unknown",
}

_ABSTENTION_PHRASES = {
    "cannot determine",
    "insufficient evidence",
    "not enough evidence",
    "unable to determine",
    "unknown",
}


@dataclass(frozen=True)
class AnswerOverlapMetrics:
    precision: float
    recall: float
    f1: float


@dataclass(frozen=True)
class RelaxedAnswerMatch:
    correct: bool
    match_type: str


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


def relaxed_answer_match(
    prediction: str | None,
    gold_answer: str | None,
) -> RelaxedAnswerMatch:
    """Return a conservative deterministic relaxed correctness diagnostic.

    This diagnostic intentionally does not replace exact match or token F1. It
    recognizes only a small set of transparent answer-format variations:

    * removable honorifics, such as ``President Richard Nixon``;
    * a month attached to a four-digit year or a unit attached to a quantity;
    * ``Party`` attached to a political label;
    * narrowly whitelisted profession domain modifiers, such as ``film director``;
    * a multi-token gold answer used as the leading or trailing answer phrase in
      a longer non-contradictory response.

    Ambiguous substring matches remain incorrect. For example, ``York`` does
    not match ``New York City``, and ``director`` does not match
    ``assistant director``.
    """

    pred = normalize_answer(prediction)
    gold = normalize_answer(gold_answer)

    if not pred or not gold:
        return RelaxedAnswerMatch(correct=False, match_type="missing_answer")

    if pred == gold:
        return RelaxedAnswerMatch(correct=True, match_type="exact")

    pred_tokens = pred.split()
    gold_tokens = gold.split()

    if _is_abstention_or_uncertain(pred, pred_tokens):
        return RelaxedAnswerMatch(correct=False, match_type="no_match")

    stripped_pred_tokens = _strip_leading_honorifics(pred_tokens)
    stripped_gold_tokens = _strip_leading_honorifics(gold_tokens)
    if stripped_pred_tokens and stripped_pred_tokens == stripped_gold_tokens:
        return RelaxedAnswerMatch(correct=True, match_type="honorific_variant")

    if _is_date_or_quantity_modifier_variant(pred_tokens, gold_tokens):
        return RelaxedAnswerMatch(correct=True, match_type="date_or_unit_modifier")

    if _is_party_variant(pred_tokens, gold_tokens):
        return RelaxedAnswerMatch(correct=True, match_type="party_modifier")

    if _is_safe_profession_modifier_variant(pred_tokens, gold_tokens):
        return RelaxedAnswerMatch(correct=True, match_type="profession_modifier")

    if _is_answer_phrase_variant(pred_tokens, gold_tokens):
        return RelaxedAnswerMatch(correct=True, match_type="answer_phrase")

    return RelaxedAnswerMatch(correct=False, match_type="no_match")


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


def _strip_leading_honorifics(tokens: list[str]) -> list[str]:
    index = 0
    while index < len(tokens) and tokens[index] in _HONORIFIC_PREFIXES:
        index += 1
    return tokens[index:]


def _is_abstention_or_uncertain(normalized_prediction: str, tokens: list[str]) -> bool:
    if any(phrase in normalized_prediction for phrase in _ABSTENTION_PHRASES):
        return True
    return bool(set(tokens) & _UNCERTAIN_OR_CONTRADICTORY_TOKENS)


def _is_date_or_quantity_modifier_variant(
    pred_tokens: list[str],
    gold_tokens: list[str],
) -> bool:
    shorter, longer = _shorter_and_longer(pred_tokens, gold_tokens)
    if len(shorter) != 1 or len(longer) != 2:
        return False

    value = shorter[0]
    if value not in longer:
        return False

    extra = longer[0] if longer[1] == value else longer[1]
    if extra in _QUANTITY_UNITS:
        return True

    return bool(re.fullmatch(r"\d{4}", value) and extra in _MONTH_NAMES)


def _is_party_variant(pred_tokens: list[str], gold_tokens: list[str]) -> bool:
    return _remove_single_trailing_token(pred_tokens, "party") == gold_tokens or (
        _remove_single_trailing_token(gold_tokens, "party") == pred_tokens
    )


def _is_safe_profession_modifier_variant(
    pred_tokens: list[str],
    gold_tokens: list[str],
) -> bool:
    shorter, longer = _shorter_and_longer(pred_tokens, gold_tokens)
    return bool(
        len(shorter) == 1
        and len(longer) == 2
        and longer[1] == shorter[0]
        and longer[0] in _SAFE_PROFESSION_MODIFIERS
    )


def _is_answer_phrase_variant(pred_tokens: list[str], gold_tokens: list[str]) -> bool:
    if len(gold_tokens) < 2 or len(pred_tokens) <= len(gold_tokens):
        return False

    if len(pred_tokens) > len(gold_tokens) + 12:
        return False

    return pred_tokens[: len(gold_tokens)] == gold_tokens or (
        pred_tokens[-len(gold_tokens) :] == gold_tokens
    )


def _remove_single_trailing_token(tokens: list[str], token: str) -> list[str]:
    if len(tokens) >= 2 and tokens[-1] == token:
        return tokens[:-1]
    return tokens


def _shorter_and_longer(
    first: list[str],
    second: list[str],
) -> tuple[list[str], list[str]]:
    if len(first) <= len(second):
        return first, second
    return second, first
