"""Exact dense search, ranking helpers and reciprocal rank fusion.

Exact, not approximate, and on purpose. An ANN index has its own recall loss, and a
recall table built on one measures two things and credits both to the embedding model.
The tune corpus fits in RAM as a flat float32 matrix with room to spare, and one query is
a single matrix-vector product, so exact search costs milliseconds here. The ANN path
belongs to serving scale, where its loss against this index gets measured as its own line.
"""

from collections.abc import Sequence

import numpy as np
import numpy.typing as npt

Ids = npt.NDArray[np.int64]
Mask = npt.NDArray[np.bool_]

# Queries scored per matrix product. 256 rows against 130k documents is about 130 MB of
# scores, which keeps the peak far under the runner's 16 GB.
QUERY_BATCH = 256

RRF_K = 60


def top_k(scores: npt.NDArray[np.float32], k: int, allowed: Mask | None = None) -> Ids:
    """Indices of the k highest scores, best first, ties broken by index.

    A tie-break matters more than it looks. Two identical chunks score identically, and
    without a fixed order the table could move between two runs of the same code.
    """
    if allowed is not None:
        scores = np.where(allowed, scores, -np.inf).astype(np.float32)
        k = min(k, int(allowed.sum()))
    k = min(k, scores.shape[0])
    if k <= 0:
        return np.zeros(0, dtype=np.int64)
    candidates = np.argpartition(-scores, k - 1)[:k]
    order = np.lexsort((candidates, -scores[candidates]))
    return candidates[order].astype(np.int64)


class FlatIndex:
    """Every row scored against every query. Rows must be unit length."""

    def __init__(self, vectors: npt.NDArray[np.float32]) -> None:
        self.vectors = np.ascontiguousarray(vectors, dtype=np.float32)

    @property
    def size(self) -> int:
        return int(self.vectors.shape[0])

    def search(
        self, queries: npt.NDArray[np.float32], k: int, allowed: Sequence[Mask] | None = None
    ) -> list[Ids]:
        """Top k per query, optionally restricted per query to the rows its mask allows.

        The mask is applied before ranking, so a restricted search is still exact over the
        rows it is allowed to see. That is the pre-filter the post-filter row is compared to.
        """
        results: list[Ids] = []
        for start in range(0, queries.shape[0], QUERY_BATCH):
            block = queries[start : start + QUERY_BATCH] @ self.vectors.T
            for offset, row in enumerate(block):
                mask = allowed[start + offset] if allowed is not None else None
                results.append(top_k(row, k, mask))
        return results


def reciprocal_rank_fusion(rankings: Sequence[Sequence[int] | Ids], k: int = RRF_K) -> list[int]:
    """Cormack et al. 2009: each list contributes 1 / (k + rank), rank counted from 1.

    Ties are broken by first appearance across the input lists, so the fused order is a
    function of the inputs alone. The dict remembers insertion order and the sort is
    stable, which is the whole of that tie-break.
    """
    fused: dict[int, float] = {}
    for ranking in rankings:
        for rank, doc in enumerate(ranking, start=1):
            fused[int(doc)] = fused.get(int(doc), 0.0) + 1.0 / (k + rank)
    return sorted(fused, key=lambda doc: -fused[doc])
