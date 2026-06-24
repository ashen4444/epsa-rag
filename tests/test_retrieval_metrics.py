import math

from epsa_rag.evaluation.retrieval_metrics import (
    both_supporting_documents_found_at_k,
    first_support_rank,
    gold_title_ranks,
    missing_gold_document_rate,
    mrr_at_k,
    ndcg_at_k,
    supporting_doc_recall_at_k,
    top1_supporting_hit,
)


def test_top1_supporting_hit_true_when_first_title_is_gold() -> None:
    retrieved = ["Marie Curie", "Radium", "Paris"]
    gold = ["Marie Curie", "Pierre Curie"]

    assert top1_supporting_hit(retrieved, gold) is True


def test_top1_supporting_hit_false_when_first_title_is_not_gold() -> None:
    retrieved = ["Radium", "Marie Curie", "Pierre Curie"]
    gold = ["Marie Curie", "Pierre Curie"]

    assert top1_supporting_hit(retrieved, gold) is False


def test_both_supporting_documents_found_at_k_true() -> None:
    retrieved = ["Noise A", "Marie Curie", "Noise B", "Pierre Curie"]
    gold = ["Marie Curie", "Pierre Curie"]

    assert both_supporting_documents_found_at_k(retrieved, gold, 4) is True


def test_both_supporting_documents_found_at_k_false_when_one_missing() -> None:
    retrieved = ["Noise A", "Marie Curie", "Noise B"]
    gold = ["Marie Curie", "Pierre Curie"]

    assert both_supporting_documents_found_at_k(retrieved, gold, 3) is False


def test_supporting_doc_recall_at_k() -> None:
    retrieved = ["Noise A", "Marie Curie", "Noise B"]
    gold = ["Marie Curie", "Pierre Curie"]

    assert supporting_doc_recall_at_k(retrieved, gold, 3) == 0.5


def test_duplicate_retrieved_titles_do_not_inflate_recall() -> None:
    retrieved = ["Marie Curie", "Marie Curie", "Marie Curie"]
    gold = ["Marie Curie", "Pierre Curie"]

    assert supporting_doc_recall_at_k(retrieved, gold, 3) == 0.5


def test_gold_title_ranks_use_first_occurrence() -> None:
    retrieved = ["Noise", "Marie Curie", "Marie Curie", "Pierre Curie"]
    gold = ["Marie Curie", "Pierre Curie"]

    assert gold_title_ranks(retrieved, gold) == {
        "Marie Curie": 2,
        "Pierre Curie": 4,
    }


def test_first_support_rank_returns_none_when_no_support_found() -> None:
    retrieved = ["Noise A", "Noise B"]
    gold = ["Marie Curie", "Pierre Curie"]

    assert first_support_rank(retrieved, gold, k=10) is None


def test_mrr_at_k() -> None:
    retrieved = ["Noise A", "Marie Curie", "Pierre Curie"]
    gold = ["Marie Curie", "Pierre Curie"]

    assert mrr_at_k(retrieved, gold, 10) == 0.5


def test_mrr_at_k_zero_when_first_support_outside_k() -> None:
    retrieved = ["Noise A", "Noise B", "Marie Curie"]
    gold = ["Marie Curie", "Pierre Curie"]

    assert mrr_at_k(retrieved, gold, 2) == 0.0


def test_ndcg_at_k_perfect_ranking() -> None:
    retrieved = ["Marie Curie", "Pierre Curie", "Noise"]
    gold = ["Marie Curie", "Pierre Curie"]

    assert ndcg_at_k(retrieved, gold, 10) == 1.0


def test_ndcg_at_k_lower_when_gold_titles_are_later() -> None:
    retrieved = ["Noise A", "Marie Curie", "Noise B", "Pierre Curie"]
    gold = ["Marie Curie", "Pierre Curie"]

    value = ndcg_at_k(retrieved, gold, 10)

    assert 0.0 < value < 1.0


def test_missing_gold_document_rate() -> None:
    retrieved = ["Marie Curie", "Noise"]
    gold = ["Marie Curie", "Pierre Curie"]

    assert missing_gold_document_rate(retrieved, gold, k=20) == 0.5


def test_title_matching_is_case_and_whitespace_insensitive() -> None:
    retrieved = ["  marie   curie  ", "PIERRE CURIE"]
    gold = ["Marie Curie", "Pierre Curie"]

    assert both_supporting_documents_found_at_k(retrieved, gold, 2) is True
    assert math.isclose(supporting_doc_recall_at_k(retrieved, gold, 2), 1.0)