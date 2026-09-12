"""The embed job's promises: an unchanged corpus costs nothing, parts reassemble into
exactly the corpus, and two jobs racing on one part leave one whole file, not a mixture."""

import hashlib
import threading
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pytest

from pr_lens.corpus.writer import LocalSink, write_units
from pr_lens.eval.corpus import load_repo
from pr_lens.eval.vectors import BODY_PREFIX, corpus_part, document_matrix, load_rows
from pr_lens.ingest.mine import review_comment_unit
from pr_lens.ingest.units import CorpusUnit
from pr_lens.jobs.embed import embed_input
from pr_lens.retrieval.embed import MODELS, EmbeddingModel, load_part

REPO = "encode/httpx"
MODEL = MODELS["bge-small"]


class HashEncoder:
    """Deterministic unit vectors derived from the text, and a count of what it encoded."""

    def __init__(self, model: EmbeddingModel = MODEL) -> None:
        self.model = model
        self.encoded = 0
        self._lock = threading.Lock()

    def encode(self, texts: Sequence[str]) -> npt.NDArray[np.float32]:
        with self._lock:
            self.encoded += len(texts)
        rows = []
        for text in texts:
            seed = int.from_bytes(hashlib.sha256(text.encode()).digest()[:8], "little")
            row = np.random.default_rng(seed).standard_normal(self.model.dims)
            rows.append(row / np.linalg.norm(row))
        return np.asarray(rows, dtype=np.float32).reshape(len(texts), self.model.dims)

    def token_counts(self, texts: Sequence[str]) -> npt.NDArray[np.int32]:
        return np.asarray([len(t.split()) for t in texts], dtype=np.int32)


def corpus_units(version: str = "1") -> list[CorpusUnit]:
    comment = review_comment_unit(
        REPO,
        {
            "id": 7,
            "path": "httpx/_client.py",
            "line": 3,
            "diff_hunk": "@@ -1,2 +1,2 @@\n-timeout = 5\n+timeout = None",
            "body": "A None timeout hangs forever on a dead server, keep a default.",
            "pull_request_url": "https://api.github.com/repos/encode/httpx/pulls/1",
        },
    )
    assert comment is not None
    return [
        CorpusUnit(
            repo=REPO,
            kind="source",
            identity=f"httpx/_client.py#f{i}#0",
            text=f"def f{i}():\n    return {i} + {version}\n",
            path="httpx/_client.py",
        )
        for i in range(20)
    ] + [comment]


@pytest.fixture
def sink(tmp_path: Path) -> LocalSink:
    store = LocalSink(tmp_path)
    write_units(store, REPO, corpus_units())
    return store


def embed_all(sink: LocalSink, parts: int, encoder: HashEncoder) -> list[bool]:
    corpus = load_repo(sink, REPO)
    return [
        embed_input(sink, corpus_part(corpus, MODEL, part, parts), encoder).skipped
        for part in range(parts)
    ]


def test_parts_reassemble_into_exactly_the_corpus_in_index_order(sink: LocalSink) -> None:
    embed_all(sink, parts=3, encoder=HashEncoder())
    corpus = load_repo(sink, REPO)
    rows = load_rows(sink, [corpus_part(corpus, MODEL, p, 3) for p in range(3)])

    with_hunk, _ = document_matrix(rows, corpus.documents, body_only=False)
    body_only, _ = document_matrix(rows, corpus.documents, body_only=True)
    expected = HashEncoder().encode([d.text for d in corpus.documents])
    assert np.array_equal(with_hunk, expected)

    comment_row = next(i for i, d in enumerate(corpus.documents) if d.body is not None)
    assert not np.array_equal(body_only[comment_row], with_hunk[comment_row])
    others = [i for i in range(len(corpus.documents)) if i != comment_row]
    assert np.array_equal(body_only[others], with_hunk[others])
    assert sum(1 for i in rows.index if i.startswith(BODY_PREFIX)) == 1


def test_an_unchanged_corpus_embeds_nothing(sink: LocalSink) -> None:
    assert embed_all(sink, parts=2, encoder=HashEncoder()) == [False, False]
    second = HashEncoder()
    assert embed_all(sink, parts=2, encoder=second) == [True, True]
    assert second.encoded == 0


def test_a_changed_corpus_is_re_embedded_and_stale_vectors_are_refused(sink: LocalSink) -> None:
    embed_all(sink, parts=1, encoder=HashEncoder())
    write_units(sink, REPO, corpus_units(version="2"))
    fresh = corpus_part(load_repo(sink, REPO), MODEL, 0, 1)

    with pytest.raises(ValueError, match="different input"):
        load_part(sink, fresh.path, fresh.fingerprint)
    assert embed_all(sink, parts=1, encoder=HashEncoder()) == [False]
    assert load_part(sink, fresh.path, fresh.fingerprint).fingerprint == fresh.fingerprint


def test_a_missing_part_is_an_error_not_a_smaller_index(sink: LocalSink) -> None:
    corpus = load_repo(sink, REPO)
    embed_input(sink, corpus_part(corpus, MODEL, 0, 2), HashEncoder())
    with pytest.raises(FileNotFoundError):
        load_rows(sink, [corpus_part(corpus, MODEL, p, 2) for p in range(2)])


def test_two_embed_passes_at_once_leave_one_whole_index(sink: LocalSink, tmp_path: Path) -> None:
    target = corpus_part(load_repo(sink, REPO), MODEL, 0, 1)
    barrier = threading.Barrier(2)
    errors: list[BaseException] = []

    def run() -> None:
        encoder = HashEncoder()
        barrier.wait()
        try:
            embed_input(sink, target, encoder)
        except BaseException as exc:
            errors.append(exc)

    threads = [threading.Thread(target=run) for _ in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert not errors

    folder = tmp_path / "eval" / "embeddings" / "bge-small" / "encode__httpx"
    assert sorted(p.name for p in folder.iterdir()) == [
        "part-00-of-01.fingerprint",
        "part-00-of-01.npz",
    ]
    stored = load_part(sink, target.path, target.fingerprint)
    assert len(set(stored.ids)) == len(stored.ids) == 22
    assert np.array_equal(stored.vectors, HashEncoder().encode(target.texts))
