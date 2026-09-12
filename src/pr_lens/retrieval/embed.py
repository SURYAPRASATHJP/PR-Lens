"""Embedding, on the Actions runner, with models pinned to an exact revision and dtype.

No hosted embedding API is a permanent free tier (checked 12 Sep 2026: HF Inference gives
a free account $0.10 a month, Jina's free key has no published renewing allowance, Cohere
and Voyage are trials). The runner is free and unlimited on a public repo, so the models
run there, and anyone who clones the repo can reproduce the table without an account.

Two models, deliberately. MAX_CHARS is 4000, roughly 1,000 to 1,300 tokens of Python, so a
512-token model reads the first part of most long chunks and none of the rest. Running a
512-token model and a long-context one over the same chunks and the same queries turns
that truncation into a number instead of a guess.
"""

import hashlib
import io
import logging
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Protocol

import numpy as np
import numpy.typing as npt

logger = logging.getLogger(__name__)

Vectors = npt.NDArray[np.float32]
TokenCounts = npt.NDArray[np.int32]


@dataclass(frozen=True, slots=True)
class EmbeddingModel:
    key: str
    name: str
    revision: str
    dims: int
    native_tokens: int
    # What this project runs it with. Equal to native_tokens unless a cost reason says
    # otherwise, and that reason is written next to the model below.
    window: int


# Revisions pinned 12 Sep 2026. A model repo can be updated in place, and a table built
# against "main" is a table nobody can rebuild.
MODELS: dict[str, EmbeddingModel] = {
    "bge-small": EmbeddingModel(
        key="bge-small",
        name="BAAI/bge-small-en-v1.5",
        revision="5c38ec7c405ec4b44b94cc5a9bb96e735b38267a",
        dims=384,
        native_tokens=512,
        window=512,
    ),
    # Run at 2,048 of its 8,192 tokens. That covers every chunk MAX_CHARS allows, which is
    # the truncation question this model is here to answer; what it cuts is the tail of
    # long issue and PR bodies. On a CPU the window is the cost: measured 12 Sep 2026 at
    # 4.2 texts/s for 395 tokens and 0.45 texts/s for 1,133, four threads, float32.
    "gte-modernbert": EmbeddingModel(
        key="gte-modernbert",
        name="Alibaba-NLP/gte-modernbert-base",
        revision="e7f32e3c00f91d699e8c43b53106206bcc72bb22",
        dims=768,
        native_tokens=8192,
        window=2048,
    ),
}


class Encoder(Protocol):
    model: EmbeddingModel

    def encode(self, texts: Sequence[str]) -> Vectors: ...

    def token_counts(self, texts: Sequence[str]) -> TokenCounts: ...


class SentenceEncoder:
    """A sentence-transformers model on the CPU, returning unit-length float32 rows.

    float32 is forced, not inherited. The gte checkpoints are stored in float16 and the
    current transformers loads a checkpoint in its stored dtype, and float16 matrix
    multiplication on a CPU without native support is emulated: measured 12 Sep 2026 at
    0.16 texts/s against 4.2 in float32 for the same 395-token input.
    """

    def __init__(self, model: EmbeddingModel, batch_size: int = 16) -> None:
        import torch
        from sentence_transformers import SentenceTransformer

        self.model = model
        self._batch_size = batch_size
        self._st = SentenceTransformer(
            model.name,
            revision=model.revision,
            device="cpu",
            model_kwargs={"dtype": torch.float32},
        )
        self._st.max_seq_length = model.window

    def encode(self, texts: Sequence[str]) -> Vectors:
        if not texts:
            return np.zeros((0, self.model.dims), dtype=np.float32)
        encoded = self._st.encode(
            list(texts),
            batch_size=self._batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return np.asarray(encoded, dtype=np.float32)

    def token_counts(self, texts: Sequence[str]) -> TokenCounts:
        """Length before truncation, special tokens included, which is what a window holds."""
        tokenizer = self._st.tokenizer
        counts = [len(ids) for ids in tokenizer(list(texts), truncation=False)["input_ids"]]
        return np.asarray(counts, dtype=np.int32)


class VectorStore(Protocol):
    def read(self, name: str) -> bytes | None: ...

    def write(self, files: Mapping[str, bytes]) -> None: ...


@dataclass(frozen=True, slots=True)
class StoredVectors:
    """Rows and the fingerprint of the input they were computed from, as one file.

    One file, so that two writers racing on the same path leave one of two complete
    results rather than the ids of one and the vectors of the other.
    """

    model: str
    fingerprint: str
    ids: tuple[str, ...]
    vectors: Vectors
    token_counts: TokenCounts

    def to_bytes(self) -> bytes:
        buffer = io.BytesIO()
        np.savez(
            buffer,
            model=np.array(self.model),
            fingerprint=np.array(self.fingerprint),
            ids=np.array(self.ids, dtype=str),
            vectors=self.vectors,
            token_counts=self.token_counts,
        )
        return buffer.getvalue()

    @classmethod
    def from_bytes(cls, payload: bytes) -> "StoredVectors":
        with np.load(io.BytesIO(payload), allow_pickle=False) as stored:
            return cls(
                model=str(stored["model"]),
                fingerprint=str(stored["fingerprint"]),
                ids=tuple(str(u) for u in stored["ids"]),
                vectors=stored["vectors"].astype(np.float32),
                token_counts=stored["token_counts"].astype(np.int32),
            )


@dataclass(frozen=True, slots=True)
class EmbedReport:
    path: str
    skipped: bool
    rows: int = 0
    seconds: float = 0.0


def part_fingerprint(source: str, model: EmbeddingModel, part: int, parts: int) -> str:
    """What a stored part is a function of. Any of these moving makes the part stale."""
    key = f"{source}|{model.name}@{model.revision}|window={model.window}|{part}/{parts}"
    return hashlib.sha256(key.encode()).hexdigest()


def embed_part(
    store: VectorStore,
    path: str,
    fingerprint: str,
    ids: Sequence[str],
    texts: Sequence[str],
    encoder: Encoder,
) -> EmbedReport:
    """Embed texts into path, unless path already holds them under this fingerprint.

    The fingerprint sits in a few bytes beside the vectors, so the skip check never
    downloads the vectors to decide. The two are written in one call, which on the Hub is
    one commit, so they can never be seen apart.
    """
    stored = store.read(f"{path}.fingerprint")
    if stored is not None and stored.decode() == fingerprint:
        logger.info("%s: unchanged, nothing to embed", path)
        return EmbedReport(path=path, skipped=True)

    started = time.monotonic()
    result = StoredVectors(
        model=encoder.model.key,
        fingerprint=fingerprint,
        ids=tuple(ids),
        vectors=encoder.encode(texts),
        token_counts=encoder.token_counts(texts),
    )
    seconds = time.monotonic() - started
    store.write({f"{path}.npz": result.to_bytes(), f"{path}.fingerprint": fingerprint.encode()})

    over = int((result.token_counts > encoder.model.window).sum())
    logger.info(
        "%s: %s texts in %.0fs (%.2f/s), %s (%.1f%%) longer than the %s-token window",
        path,
        len(texts),
        seconds,
        len(texts) / max(seconds, 1e-9),
        over,
        100.0 * over / max(len(texts), 1),
        encoder.model.window,
    )
    return EmbedReport(path=path, skipped=False, rows=len(texts), seconds=seconds)


def load_part(store: VectorStore, path: str, fingerprint: str) -> StoredVectors:
    """The stored rows, refusing any computed from input other than what is expected."""
    payload = store.read(f"{path}.npz")
    if payload is None:
        raise FileNotFoundError(f"{path}.npz is missing. Run the embed job for it first.")
    stored = StoredVectors.from_bytes(payload)
    if stored.fingerprint != fingerprint:
        raise ValueError(
            f"{path} was embedded from different input, or with a different model, window "
            "or part count. Re-run the embed job; it redoes exactly the stale parts."
        )
    if not np.isfinite(stored.vectors).all():
        raise ValueError(f"{path} contains non-finite values")
    return stored
