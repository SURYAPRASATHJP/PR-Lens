"""BM25 with a tokenizer that knows what an identifier is.

Whitespace tokenizing code turns get_app_dir(ctx) into one token that matches nothing.
Here an identifier yields itself and its parts, so get_app_dir also matches app_dir, and
parseHeaderLine also matches header. Written out rather than taken from a library because
the tokenizer is the part that matters, and the scoring is twenty lines of sparse algebra.
"""

import re
from collections import Counter
from collections.abc import Sequence

import numpy as np
import numpy.typing as npt
from scipy import sparse

_IDENTIFIER = re.compile(r"[A-Za-z_][A-Za-z0-9_]*|\d+")
_CAMEL = re.compile(r"[A-Z]+(?=[A-Z][a-z])|[A-Z]?[a-z]+|[A-Z]+|\d+")

Scores = npt.NDArray[np.float32]


def tokenize(text: str) -> list[str]:
    tokens = []
    for identifier in _IDENTIFIER.findall(text):
        whole = identifier.lower().strip("_")
        if not whole:
            continue
        tokens.append(whole)
        parts = [p.lower() for piece in identifier.split("_") for p in _CAMEL.findall(piece)]
        if len(parts) > 1:
            tokens.extend(parts)
    return tokens


class BM25Index:
    """Okapi BM25, k1 = 1.2 and b = 0.75, with the non-negative Lucene idf.

    The document side is precomputed into one sparse matrix of per-term weights, so a
    query is a sum of the columns for its distinct terms. Distinct, because a hunk that
    names `self` forty times is not forty times more about `self`.
    """

    def __init__(self, texts: Sequence[str], k1: float = 1.2, b: float = 0.75) -> None:
        self.vocabulary: dict[str, int] = {}
        rows: list[int] = []
        cols: list[int] = []
        counts: list[int] = []
        lengths = np.zeros(len(texts), dtype=np.float32)
        for row, text in enumerate(texts):
            terms = Counter(tokenize(text))
            lengths[row] = sum(terms.values())
            for term, count in terms.items():
                rows.append(row)
                cols.append(self.vocabulary.setdefault(term, len(self.vocabulary)))
                counts.append(count)

        n = len(texts)
        tf = sparse.csr_matrix(
            (np.asarray(counts, dtype=np.float32), (rows, cols)),
            shape=(n, len(self.vocabulary)),
        )
        df = np.bincount(np.asarray(cols, dtype=np.int64), minlength=len(self.vocabulary))
        idf = np.log1p((n - df + 0.5) / (df + 0.5)).astype(np.float32)

        average = float(lengths.mean()) if n else 0.0
        norm = k1 * (1.0 - b + b * lengths / max(average, 1e-9))
        # Per nonzero entry: tf * (k1 + 1) / (tf + norm(row)), then scaled by idf(column).
        row_of_entry = np.repeat(np.arange(n), np.diff(tf.indptr))
        tf.data = tf.data * (k1 + 1.0) / (tf.data + norm[row_of_entry])
        self._weights = sparse.csc_matrix(tf @ sparse.diags(idf))
        self.size = n

    def scores(self, query: str) -> Scores:
        terms = {self.vocabulary[t] for t in tokenize(query) if t in self.vocabulary}
        if not terms:
            return np.zeros(self.size, dtype=np.float32)
        column_sum = self._weights[:, sorted(terms)].sum(axis=1)
        return np.asarray(column_sum, dtype=np.float32).ravel()
