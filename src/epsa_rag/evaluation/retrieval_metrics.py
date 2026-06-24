from __future__ import annotations

import math
import re
import unicodedata
from typing import Iterable, Sequence


def normalize_title(title: str | None) -> str:
    """Normalize document titles for stable title-level matching."""
    if title is None:
        return ""

    normalized = unicodedata.normalize("NFKC", title)
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return normalized.casefold()


def unique_normalized_titles(titles: Iterable[str | None]) -> list[str]:
    """Return normalized titles in first-seen order without duplicates."""
    seen: set[str] = set()
    unique_titles: list[str] = []

    for title in titles:
        normalized = normalize_title(title)
        if not normalized:
            continue

        if normalized not in seen:
            seen.add(normalized)
            unique_titles.append(normalized)

    return unique_titles


def top1_supporting_hit(
    retrieved_titles: Sequence[str],
    gold_supporting_titles: Sequence[str],
) -> bool:
    """Return True if rank-1 retrieved title is one of the current question's gold titles."""
    if not retrieved_titles:
        return False

    gold_set = set(unique_normalized_titles(gold_supporting_titles))
    if not gold_set:
        return False

    return normalize_title(retrieved_titles[0]) in gold_set


def supporting_doc_recall_at_k(
    retrieved_titles: Sequence[str],
    gold_supporting_titles: Sequence[str],
    k: int,
) -> float:
    """Compute title-level supporting document recall@k.

    Duplicate retrieved titles do not inflate recall.
    """
    if k <= 0:
        return 0.0

    gold_set = set(unique_normalized_titles(gold_supporting_titles))
    if not gold_set:
        return 0.0

    retrieved_set = set(unique_normalized_titles(retrieved_titles[:k]))
    found_count = len(gold_set.intersection(retrieved_set))

    return found_count / len(gold_set)


def both_supporting_documents_found_at_k(
    retrieved_titles: Sequence[str],
    gold_supporting_titles: Sequence[str],
    k: int,
) -> bool:
    """Return True if both HotPotQA gold supporting document titles are found within top-k.

    This project stage assumes exactly two gold supporting documents per valid question.
    """
    if k <= 0:
        return False

    gold_set = set(unique_normalized_titles(gold_supporting_titles))
    if len(gold_set) != 2:
        return False

    retrieved_set = set(unique_normalized_titles(retrieved_titles[:k]))
    return gold_set.issubset(retrieved_set)


def gold_title_ranks(
    retrieved_titles: Sequence[str],
    gold_supporting_titles: Sequence[str],
) -> dict[str, int | None]:
    """Return first 1-based rank for each gold supporting title.

    Duplicate retrieved titles are ignored after their first occurrence.
    """
    gold_titles_clean = [title for title in gold_supporting_titles if normalize_title(title)]
    gold_norm_to_original: dict[str, str] = {
        normalize_title(title): title for title in gold_titles_clean
    }

    ranks: dict[str, int | None] = {
        original_title: None for original_title in gold_norm_to_original.values()
    }

    seen_retrieved_titles: set[str] = set()

    for index, retrieved_title in enumerate(retrieved_titles, start=1):
        normalized = normalize_title(retrieved_title)
        if not normalized or normalized in seen_retrieved_titles:
            continue

        seen_retrieved_titles.add(normalized)

        if normalized in gold_norm_to_original:
            original_gold_title = gold_norm_to_original[normalized]
            if ranks[original_gold_title] is None:
                ranks[original_gold_title] = index

    return ranks


def first_support_rank(
    retrieved_titles: Sequence[str],
    gold_supporting_titles: Sequence[str],
    k: int | None = None,
) -> int | None:
    """Return first 1-based rank of any gold supporting document."""
    if k is None:
        search_titles = retrieved_titles
    else:
        search_titles = retrieved_titles[:k]

    gold_set = set(unique_normalized_titles(gold_supporting_titles))
    if not gold_set:
        return None

    seen_retrieved_titles: set[str] = set()

    for index, retrieved_title in enumerate(search_titles, start=1):
        normalized = normalize_title(retrieved_title)
        if not normalized or normalized in seen_retrieved_titles:
            continue

        seen_retrieved_titles.add(normalized)

        if normalized in gold_set:
            return index

    return None


def mrr_at_k(
    retrieved_titles: Sequence[str],
    gold_supporting_titles: Sequence[str],
    k: int,
) -> float:
    """Compute reciprocal rank of first supporting document within top-k."""
    rank = first_support_rank(retrieved_titles, gold_supporting_titles, k=k)
    if rank is None:
        return 0.0

    return 1.0 / rank


def ndcg_at_k(
    retrieved_titles: Sequence[str],
    gold_supporting_titles: Sequence[str],
    k: int,
) -> float:
    """Compute title-level binary nDCG@k.

    A gold document receives relevance 1 once, at its first retrieved occurrence.
    Duplicate retrieved titles receive relevance 0 after the first occurrence.
    """
    if k <= 0:
        return 0.0

    gold_set = set(unique_normalized_titles(gold_supporting_titles))
    if not gold_set:
        return 0.0

    seen_retrieved_titles: set[str] = set()
    dcg = 0.0

    for rank, retrieved_title in enumerate(retrieved_titles[:k], start=1):
        normalized = normalize_title(retrieved_title)
        if not normalized or normalized in seen_retrieved_titles:
            relevance = 0
        else:
            relevance = 1 if normalized in gold_set else 0
            seen_retrieved_titles.add(normalized)

        if relevance:
            dcg += relevance / math.log2(rank + 1)

    ideal_relevant_count = min(len(gold_set), k)
    ideal_dcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_relevant_count + 1))

    if ideal_dcg == 0:
        return 0.0

    return dcg / ideal_dcg


def mean_found_gold_rank(
    retrieved_titles: Sequence[str],
    gold_supporting_titles: Sequence[str],
) -> float | None:
    """Return the mean rank of found gold supporting documents."""
    ranks = gold_title_ranks(retrieved_titles, gold_supporting_titles)
    found_ranks = [rank for rank in ranks.values() if rank is not None]

    if not found_ranks:
        return None

    return sum(found_ranks) / len(found_ranks)


def missing_gold_document_rate(
    retrieved_titles: Sequence[str],
    gold_supporting_titles: Sequence[str],
    k: int | None = None,
) -> float:
    """Return fraction of gold supporting titles missing from retrieved results."""
    gold_set = set(unique_normalized_titles(gold_supporting_titles))
    if not gold_set:
        return 0.0

    if k is None:
        considered_titles = retrieved_titles
    else:
        considered_titles = retrieved_titles[:k]

    retrieved_set = set(unique_normalized_titles(considered_titles))
    missing_count = len(gold_set.difference(retrieved_set))

    return missing_count / len(gold_set)


def approximate_token_count(text: str | None) -> int:
    """Approximate tokens using a simple character-based estimate.

    This avoids requiring a tokenizer during retrieval evaluation.
    """
    if not text:
        return 0

    return max(1, math.ceil(len(text) / 4))