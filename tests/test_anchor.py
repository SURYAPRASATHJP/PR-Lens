import math

import pytest

from pr_lens.jobs.anchor import ndcg_at_k, sample


def test_ndcg_is_one_when_the_only_relevant_document_is_first() -> None:
    assert ndcg_at_k(["d1", "d2", "d3"], {"d1": 1}) == 1.0


def test_ndcg_discounts_by_log_rank_with_linear_gain() -> None:
    assert ndcg_at_k(["d9", "d1"], {"d1": 1}) == pytest.approx(1 / math.log2(3))
    # Graded relevance, as trec_eval's ndcg_cut computes it: the gain is the grade itself.
    graded = ndcg_at_k(["a", "b"], {"a": 1, "b": 2})
    ideal = 2 / math.log2(2) + 1 / math.log2(3)
    assert graded == pytest.approx((1 / math.log2(2) + 2 / math.log2(3)) / ideal)


def test_ndcg_ignores_anything_past_the_cutoff() -> None:
    ranked = [f"x{i}" for i in range(10)] + ["d1"]
    assert ndcg_at_k(ranked, {"d1": 1}) == 0.0


def test_the_query_sample_is_stable_and_not_the_first_n() -> None:
    ids = [str(i) for i in range(5000)]
    assert sample(ids, 100) == sample(list(reversed(ids)), 100)
    assert sample(ids, 100) != sorted(ids)[:100]
    assert len(set(sample(ids, 100))) == 100
