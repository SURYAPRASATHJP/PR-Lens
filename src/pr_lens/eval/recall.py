"""The arithmetic of the recall table, kept apart from the models and the storage.

Every number in the table comes from here, computed from rankings and a known gold row.
Nothing is judged: each query has exactly one correct answer, the review comment a human
left on that hunk, so recall@k is the fraction of queries whose gold is in the top k.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt

from pr_lens.eval.corpus import EvalDocument
from pr_lens.eval.pairs import QUERY_VARIANTS, ReviewPair, query_text
from pr_lens.retrieval.search import FlatIndex, Ids, Mask

KS = (1, 5, 10, 20, 50)

# Each first stage returns this many, so fusion has room below the deepest k reported.
FUSION_DEPTH = 100
RERANK_DEPTH = 30
PLATEAU_DEPTHS = (5, 10, 15, 20, 25, 30)

# Candidates fetched before a post-filter. 40 is pgvector's default hnsw.ef_search, which
# is exactly this situation at serving time: the index returns its candidates and the
# WHERE clause is applied to whatever came back.
POST_FILTER_CANDIDATES = (10, 40, 100, 400)
POST_FILTER_K = 10

# The dense index the table is measured on. Asserted by a test, so an approximate index
# cannot be swapped in later without the table saying it measures something else.
DENSE_INDEX = FlatIndex

_NOT_FOUND = np.iinfo(np.int64).max

_LANGUAGES = {
    ".py": "python",
    ".pyi": "python",
    ".md": "docs",
    ".rst": "docs",
    ".txt": "docs",
    ".mdx": "docs",
    ".markdown": "docs",
    ".toml": "config",
    ".cfg": "config",
    ".ini": "config",
}


class LeakageError(AssertionError):
    """A gold document contains its own query, so the row would measure string matching."""


def assert_no_leakage(pairs: Sequence[ReviewPair], gold_texts: Mapping[str, str]) -> None:
    """Run on the real pair set before a body-only number is computed, every time.

    The body-only row is the one that goes in the README, and it is only honest if no gold
    carries its query. A test covers the serialisation; this covers the data.
    """
    offenders = []
    for pair in pairs:
        gold = gold_texts[pair.gold_unit_id]
        for variant in QUERY_VARIANTS:
            # Stripped, because a hunk's trailing newline would otherwise hide a real match.
            query = query_text(pair, variant).strip()
            if query and query in gold:
                offenders.append(f"{pair.repo} comment {pair.comment_id} ({variant})")
    if offenders:
        raise LeakageError(
            f"{len(offenders)} body-only gold documents contain their query: "
            + ", ".join(offenders[:10])
        )


def language(path: str | None) -> str:
    if not path or "." not in path.rsplit("/", 1)[-1]:
        return "none"
    return _LANGUAGES.get("." + path.rsplit(".", 1)[-1].lower(), "other")


@dataclass(frozen=True, slots=True)
class DocumentTable:
    """The documents in index order, and the columns a filter needs, as arrays."""

    documents: tuple[EvalDocument, ...]
    row_of: Mapping[str, int]
    repo: npt.NDArray[np.str_]
    kind: npt.NDArray[np.str_]
    language: npt.NDArray[np.str_]

    @classmethod
    def build(cls, documents: Sequence[EvalDocument]) -> "DocumentTable":
        return cls(
            documents=tuple(documents),
            row_of={d.unit_id: i for i, d in enumerate(documents)},
            repo=np.array([d.repo for d in documents]),
            kind=np.array([d.kind for d in documents]),
            language=np.array([language(d.path) for d in documents]),
        )

    def gold_rows(self, pairs: Sequence[ReviewPair]) -> npt.NDArray[np.int64]:
        return np.array([self.row_of[p.gold_unit_id] for p in pairs], dtype=np.int64)


def gold_ranks(rankings: Sequence[Ids], gold: npt.NDArray[np.int64]) -> npt.NDArray[np.int64]:
    """Zero-based rank of the gold in each ranking, or a sentinel when it is absent."""
    ranks = np.full(len(rankings), _NOT_FOUND, dtype=np.int64)
    for i, (ranking, target) in enumerate(zip(rankings, gold, strict=True)):
        hits = np.flatnonzero(np.asarray(ranking) == target)
        if hits.size:
            ranks[i] = int(hits[0])
    return ranks


def recall_at(ranks: npt.NDArray[np.int64], ks: Sequence[int] = KS) -> dict[int, float]:
    if ranks.size == 0:
        return dict.fromkeys(ks, 0.0)
    return {k: float((ranks < k).mean()) for k in ks}


def interval(p: float, n: int) -> float:
    """Half-width of a 95% Wald interval, so a reader can tell a gap from noise."""
    return 1.96 * float(np.sqrt(max(p * (1.0 - p), 0.0) / max(n, 1)))


@dataclass(frozen=True, slots=True)
class PostFilterRow:
    candidates: int | None  # None is the pre-filter: exact search over allowed rows only
    recall: float
    mean_returned: float
    short_share: float  # fraction of queries given fewer than POST_FILTER_K results


def post_filter(
    index: FlatIndex,
    query_vectors: npt.NDArray[np.float32],
    gold: npt.NDArray[np.int64],
    masks: Sequence[Mask],
) -> list[PostFilterRow]:
    """The same filtered question answered two ways.

    Pre-filter is the right answer: search only the rows the filter allows, exactly. Post-
    filter is what an index that cannot filter does: fetch C candidates, then drop the ones
    the filter rejects. The gold always passes these filters, so every loss below is the
    candidate budget running out before the filtered set does.
    """
    rows = []
    pre = index.search(query_vectors, POST_FILTER_K, allowed=masks)
    rows.append(_filter_row(None, pre, gold))
    deepest = index.search(query_vectors, max(POST_FILTER_CANDIDATES))
    for budget in POST_FILTER_CANDIDATES:
        survivors = [
            ranking[:budget][mask[ranking[:budget]]][:POST_FILTER_K]
            for ranking, mask in zip(deepest, masks, strict=True)
        ]
        rows.append(_filter_row(budget, survivors, gold))
    return rows


def _filter_row(
    budget: int | None, rankings: Sequence[Ids], gold: npt.NDArray[np.int64]
) -> PostFilterRow:
    lengths = np.array([len(r) for r in rankings])
    return PostFilterRow(
        candidates=budget,
        recall=recall_at(gold_ranks(rankings, gold), (POST_FILTER_K,))[POST_FILTER_K],
        mean_returned=float(lengths.mean()) if lengths.size else 0.0,
        short_share=float((lengths < POST_FILTER_K).mean()) if lengths.size else 0.0,
    )


@dataclass(frozen=True, slots=True)
class PlateauRow:
    depth: int
    first_stage: float  # recall@depth before reranking, the ceiling reranking cannot pass
    reranked: dict[int, float] = field(default_factory=dict)


def rerank_plateau(
    candidates: Sequence[Ids],
    scores: Sequence[npt.NDArray[np.float32]],
    gold: npt.NDArray[np.int64],
    depths: Sequence[int] = PLATEAU_DEPTHS,
    ks: Sequence[int] = (1, 5, 10),
) -> list[PlateauRow]:
    """Recall after the cross-encoder reorders only the first `depth` candidates.

    Each query's candidates are scored once, to the full rerank depth, and every smaller
    depth reuses a prefix of those scores. Cross-encoder scores are per pair, so reranking
    the top 10 is exactly the top 10 of the top-30 scores, and nothing is re-run.
    """
    rows = []
    for depth in depths:
        reordered = []
        for ranking, score in zip(candidates, scores, strict=True):
            head = np.asarray(ranking[:depth])
            order = np.argsort(-score[: len(head)], kind="stable")
            reordered.append(head[order])
        ranks = gold_ranks(reordered, gold)
        rows.append(
            PlateauRow(
                depth=depth,
                first_stage=recall_at(gold_ranks([c[:depth] for c in candidates], gold), (depth,))[
                    depth
                ],
                reranked={k: v for k, v in recall_at(ranks, ks).items() if k <= depth},
            )
        )
    return rows


def length_buckets(
    ranks: npt.NDArray[np.int64], query_tokens: npt.NDArray[np.int32], window: int, k: int = 10
) -> dict[str, tuple[int, float]]:
    """recall@k for queries that fit the window and queries that do not.

    The truncation price, measured where it is paid: a 512-token model reads the first 512
    tokens of a long hunk and none of the rest.
    """
    fits = query_tokens <= window
    return {
        f"<= {window} tokens": (int(fits.sum()), recall_at(ranks[fits], (k,))[k]),
        f"> {window} tokens": (int((~fits).sum()), recall_at(ranks[~fits], (k,))[k]),
    }
