import math
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pytest

from pr_lens.corpus.writer import LocalSink
from pr_lens.jobs import anchor
from pr_lens.jobs.anchor import fingerprint, ndcg_at_k, part_path, sample
from pr_lens.jobs.plan import ANCHOR_PARTS
from pr_lens.retrieval.embed import load_part


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


# What the first Actions run stored beside its eight anchor parts, computed from the code at
# 60e6b81 that produced them. If this fails, the next run re-embeds all 280,310 documents,
# about 76 minutes on eight runners. Change these only on purpose.
STORED_FINGERPRINTS = [
    "7c4e889016e3f458e94be0300e77889e3addec96bc417a8c4ff04dd6be2203bb",
    "25c5ea67c1beae0db4b214ca9555d8b7e6c89662dbf4b775e7fdccae722c9fba",
    "36c1fc2fe454c605d980ed1481329c0098da44ad8c6bfec54b2c24698e230894",
    "796edb2f50492a47c694a637d0e70bd4b2512ab0a78c2fff96c287473db51f49",
    "7ca572d9705e4cd914ceede9a35600e011d161433a888dca195a592388f4c4a1",
    "e19cf83d204f8ca4bd18dbed3b8dc9553a935b3a7e8259ecd75caef6d5a6e557",
    "15c61435d22df36a7e143925fd0e7286e096fc09ca2e80070ced0071fe791163",
    "3fce00af1a02477c26a23473b44f59786d2be06121d6ae9685e00b1639e5f471",
]


def test_the_stored_anchor_parts_are_still_current() -> None:
    assert len(STORED_FINGERPRINTS) == ANCHOR_PARTS
    assert [fingerprint(p, ANCHOR_PARTS) for p in range(ANCHOR_PARTS)] == STORED_FINGERPRINTS


def test_a_finished_part_is_skipped_before_any_download_or_model_load(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def must_not_run(*_: object) -> None:
        raise AssertionError("a finished part must not download the corpus or load a model")

    monkeypatch.setattr(anchor, "corpus", must_not_run)
    monkeypatch.setattr(anchor, "SentenceEncoder", must_not_run)
    store = LocalSink(tmp_path)
    store.write({f"{part_path(3, ANCHOR_PARTS)}.fingerprint": STORED_FINGERPRINTS[3].encode()})

    anchor.embed(store, 3, ANCHOR_PARTS)


def test_an_unfinished_part_is_embedded_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    loads: list[str] = []

    class Encoder:
        model = anchor.MODEL

        def __init__(self, model: object) -> None:
            loads.append("model")

        def encode(self, texts: Sequence[str]) -> npt.NDArray[np.float32]:
            return np.ones((len(texts), self.model.dims), dtype=np.float32)

        def token_counts(self, texts: Sequence[str]) -> npt.NDArray[np.int32]:
            return np.ones(len(texts), dtype=np.int32)

    monkeypatch.setattr(anchor, "corpus", lambda: ([f"d{i}" for i in range(16)], ["doc"] * 16))
    monkeypatch.setattr(anchor, "SentenceEncoder", Encoder)
    store = LocalSink(tmp_path)

    anchor.embed(store, 1, ANCHOR_PARTS)
    anchor.embed(store, 1, ANCHOR_PARTS)

    assert loads == ["model"]
    stored = load_part(store, part_path(1, ANCHOR_PARTS), fingerprint(1, ANCHOR_PARTS))
    assert stored.ids == ("d1", "d9")
