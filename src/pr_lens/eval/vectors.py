"""Which texts each embedding part holds, and how the parts come back together.

A part is a deterministic slice, every n-th item from a fixed order, so the same part
number always means the same texts and a matrix job can be re-run on its own. Reassembly
checks that the parts cover exactly the expected ids, so a missing or doubled part is an
error rather than a smaller or skewed index.
"""

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from pr_lens.corpus.writer import repo_slug
from pr_lens.eval.corpus import EvalDocument, RepoCorpus
from pr_lens.eval.pairs import QUERY_VARIANTS, QueryVariant, ReviewPair, query_text
from pr_lens.retrieval.embed import (
    EmbeddingModel,
    StoredVectors,
    TokenCounts,
    Vectors,
    VectorStore,
    load_part,
    part_fingerprint,
)

BODY_PREFIX = "body/"


@dataclass(frozen=True, slots=True)
class PartInput:
    path: str
    fingerprint: str
    ids: tuple[str, ...]
    texts: tuple[str, ...]


def corpus_part(corpus: RepoCorpus, model: EmbeddingModel, part: int, parts: int) -> PartInput:
    """Every unit's indexed text, plus the body-only text of each review comment in it."""
    _check_part(part, parts)
    documents = corpus.documents[part::parts]
    ids = [d.unit_id for d in documents]
    texts = [d.text for d in documents]
    for document in documents:
        if document.body is not None:
            ids.append(BODY_PREFIX + document.unit_id)
            texts.append(document.body)
    return PartInput(
        path=corpus_part_path(model.key, corpus.repo, part, parts),
        fingerprint=part_fingerprint(corpus.fingerprint, model, part, parts),
        ids=tuple(ids),
        texts=tuple(texts),
    )


def query_part(
    pairs: Sequence[ReviewPair],
    pairs_digest: str,
    version: str,
    model: EmbeddingModel,
    part: int,
    parts: int,
) -> PartInput:
    _check_part(part, parts)
    ids = []
    texts = []
    for pair in pairs[part::parts]:
        for variant in QUERY_VARIANTS:
            ids.append(query_id(pair, variant))
            texts.append(query_text(pair, variant))
    return PartInput(
        path=query_part_path(version, model.key, part, parts),
        fingerprint=part_fingerprint(pairs_digest, model, part, parts),
        ids=tuple(ids),
        texts=tuple(texts),
    )


def query_id(pair: ReviewPair, variant: QueryVariant) -> str:
    return f"{pair.repo}#{pair.comment_id}/{variant}"


def corpus_part_path(model_key: str, repo: str, part: int, parts: int) -> str:
    return f"eval/embeddings/{model_key}/{repo_slug(repo)}/part-{part:02d}-of-{parts:02d}"


def query_part_path(version: str, model_key: str, part: int, parts: int) -> str:
    return f"eval/queries/{version}/{model_key}/part-{part:02d}-of-{parts:02d}"


@dataclass(frozen=True, slots=True)
class Rows:
    """Vectors and token counts keyed by id, from any number of parts."""

    index: dict[str, int]
    vectors: Vectors
    token_counts: TokenCounts

    @classmethod
    def join(cls, parts: Sequence[StoredVectors]) -> "Rows":
        ids = [i for part in parts for i in part.ids]
        index = {identifier: row for row, identifier in enumerate(ids)}
        if len(index) != len(ids):
            raise ValueError("two embedding parts hold the same id")
        return cls(
            index=index,
            vectors=np.concatenate([p.vectors for p in parts])
            if parts
            else np.zeros((0, 0), np.float32),
            token_counts=np.concatenate([p.token_counts for p in parts])
            if parts
            else np.zeros(0, np.int32),
        )

    def take(self, ids: Sequence[str]) -> tuple[Vectors, TokenCounts]:
        missing = [i for i in ids if i not in self.index]
        if missing:
            raise KeyError(f"{len(missing)} ids have no vector, first {missing[0]}")
        rows = np.array([self.index[i] for i in ids], dtype=np.int64)
        return self.vectors[rows], self.token_counts[rows]


def load_rows(store: VectorStore, inputs: Sequence[PartInput]) -> Rows:
    """Every part, with each checked against the fingerprint it should have been built from."""
    rows = Rows.join([load_part(store, p.path, p.fingerprint) for p in inputs])
    expected = {i for p in inputs for i in p.ids}
    if set(rows.index) != expected:
        raise ValueError("the embedding parts do not cover exactly the expected texts")
    return rows


def document_matrix(
    rows: Rows, documents: Sequence[EvalDocument], body_only: bool
) -> tuple[Vectors, TokenCounts]:
    """The index matrix in document order, with comments swapped to their body when asked."""
    ids = [
        BODY_PREFIX + d.unit_id if body_only and d.body is not None else d.unit_id
        for d in documents
    ]
    return rows.take(ids)


def _check_part(part: int, parts: int) -> None:
    if parts < 1 or not 0 <= part < parts:
        raise ValueError(f"part {part} of {parts} does not exist")
