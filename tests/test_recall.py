import json
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pytest

from pr_lens.eval import recall
from pr_lens.eval.corpus import body_only
from pr_lens.eval.pairs import ReviewPair, screen_comment
from pr_lens.eval.recall import (
    POST_FILTER_CANDIDATES,
    LeakageError,
    assert_no_leakage,
    gold_ranks,
    post_filter,
    recall_at,
    rerank_plateau,
)
from pr_lens.ingest.mine import review_comment_unit
from pr_lens.retrieval.bm25 import BM25Index, tokenize
from pr_lens.retrieval.search import FlatIndex, reciprocal_rank_fusion, top_k


def unit_rows(n: int, dims: int, seed: int = 0) -> npt.NDArray[np.float32]:
    rows = np.random.default_rng(seed).standard_normal((n, dims)).astype(np.float32)
    return rows / np.linalg.norm(rows, axis=1, keepdims=True)


def test_the_table_is_measured_on_exact_search() -> None:
    """Swapping in an ANN index would change what every dense row measures."""
    assert recall.DENSE_INDEX is FlatIndex


def test_flat_search_is_exactly_the_brute_force_ranking() -> None:
    docs, queries = unit_rows(2000, 32, seed=1), unit_rows(300, 32, seed=2)
    found = FlatIndex(docs).search(queries, 50)
    scores = queries @ docs.T
    for row, ids in zip(scores, found, strict=True):
        expected = np.lexsort((np.arange(row.size), -row))[:50]
        assert np.array_equal(ids, expected)


def test_ties_are_broken_by_row_so_a_rerun_cannot_move_the_table() -> None:
    scores = np.array([0.5, 0.9, 0.9, 0.1, 0.9], dtype=np.float32)
    assert top_k(scores, 3).tolist() == [1, 2, 4]


def test_a_masked_search_is_still_exact_over_what_it_may_see() -> None:
    docs, queries = unit_rows(500, 16, seed=3), unit_rows(5, 16, seed=4)
    mask = np.zeros(500, dtype=bool)
    mask[::7] = True
    found = FlatIndex(docs).search(queries, 10, allowed=[mask] * 5)
    for query, ids in zip(queries, found, strict=True):
        allowed = np.flatnonzero(mask)
        scores = docs[allowed] @ query
        assert np.array_equal(ids, allowed[np.lexsort((allowed, -scores))[:10]])


def test_recall_counts_the_gold_inside_the_cutoff() -> None:
    rankings = [np.array([3, 1, 2]), np.array([9, 8, 7]), np.array([5, 4, 0])]
    ranks = gold_ranks(rankings, np.array([1, 0, 0]))
    assert recall_at(ranks, (1, 2, 3)) == {1: 0.0, 2: 1 / 3, 3: 2 / 3}


def test_rrf_rewards_agreement_between_the_two_lists() -> None:
    fused = reciprocal_rank_fusion([[1, 2, 3], [3, 4, 1]])
    assert fused[:2] == [1, 3]
    assert set(fused) == {1, 2, 3, 4}


def test_the_tokenizer_splits_identifiers_into_their_parts() -> None:
    tokens = tokenize("def get_app_dir(parseHeaderLine): return HTTPServer")
    for token in ("get_app_dir", "app", "dir", "parseheaderline", "header", "http", "server"):
        assert token in tokens


def test_bm25_ranks_the_document_that_shares_the_rare_terms() -> None:
    index = BM25Index(
        [
            "the cache directory is created lazily",
            "timeout defaults to five seconds in the client",
            "the the the the",
        ]
    )
    scores = index.scores("what is the client timeout default")
    assert int(np.argmax(scores)) == 1
    assert index.scores("nonexistent vocabulary").sum() == 0


def test_post_filtering_loses_what_pre_filtering_keeps() -> None:
    # The gold sits at unfiltered rank 60, behind 59 closer documents that the filter
    # rejects. Pre-filtering finds it first; post-filtering finds it only once the
    # candidate budget reaches past rank 60.
    dims = 8
    query = np.zeros((1, dims), dtype=np.float32)
    query[0, 0] = 1.0
    docs = np.zeros((500, dims), dtype=np.float32)
    closeness = np.linspace(0.99, 0.01, 500)
    docs[:, 0] = closeness
    docs[:, 1] = np.sqrt(1 - closeness**2)
    mask = np.zeros(500, dtype=bool)
    mask[60:] = True
    gold = np.array([60])

    rows = {row.candidates: row for row in post_filter(FlatIndex(docs), query, gold, [mask])}

    assert rows[None].recall == 1.0
    assert rows[10].recall == 0.0 and rows[10].mean_returned == 0.0
    assert rows[40].recall == 0.0
    assert rows[100].recall == 1.0
    assert set(rows) == {None, *POST_FILTER_CANDIDATES}


def test_the_plateau_reranks_prefixes_of_one_scoring_pass() -> None:
    candidates = [np.arange(30)]
    scores = [np.linspace(0, 1, 30)[::-1].copy().astype(np.float32)]
    scores[0][20] = 5.0  # the cross-encoder loves the gold, which the first stage put 21st
    rows = {row.depth: row for row in rerank_plateau(candidates, scores, np.array([20]))}

    assert rows[10].first_stage == 0.0 and rows[10].reranked[1] == 0.0
    assert rows[25].first_stage == 1.0 and rows[25].reranked[1] == 1.0
    assert 10 not in rows[5].reranked


FIXTURES = json.loads(
    (Path(__file__).parent / "fixtures" / "review_comments.json").read_text(encoding="utf-8")
)["comments"]


def fixture_pairs() -> list[tuple[ReviewPair, dict[str, str]]]:
    out = []
    for case in FIXTURES:
        screened = screen_comment(case["payload"])
        if not isinstance(screened, ReviewPair):
            continue
        pair = ReviewPair(**{**screened.as_record(), "repo": "fastapi/typer"})
        unit = review_comment_unit("fastapi/typer", case["payload"])
        assert unit is not None
        out.append(
            (
                pair,
                {"with_hunk": unit.text, "body_only": body_only(unit.as_record())},
            )
        )
    return out


def test_no_body_only_gold_contains_its_query() -> None:
    """The leakage rule, enforced: the body-only gold never carries the hunk it answers."""
    pairs = fixture_pairs()
    assert pairs
    assert_no_leakage(
        [p for p, _ in pairs], {p.gold_unit_id: texts["body_only"] for p, texts in pairs}
    )


def test_the_with_hunk_gold_does_contain_its_query_so_the_check_can_fire() -> None:
    pairs = fixture_pairs()
    with pytest.raises(LeakageError, match="contain their query"):
        assert_no_leakage(
            [p for p, _ in pairs], {p.gold_unit_id: texts["with_hunk"] for p, texts in pairs}
        )
