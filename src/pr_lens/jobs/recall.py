"""The Phase 2 recall table, in three steps that the workflow runs as separate jobs.

first-stage   dense, BM25 and RRF for both models, both serialisations and both query
              variants; the truncation numbers; the post-filter rows; latency; and the
              candidate lists for the rerank sample.
rerank        cross-encoder scores for one shard of the rerank sample. The reranker is
              the slowest thing in the phase, so it is a matrix.
table         the plateau from the rerank shards, and the whole table as markdown.

Each step reads the corpus and the frozen query set from the dataset repo and writes its
result back there, under the query set's version, so a step can be re-run on its own.
"""

import argparse
import asyncio
import hashlib
import json
import logging
import os
import platform
import statistics
import sys
import time
from collections import Counter
from collections.abc import Callable, Iterable, Sequence
from dataclasses import asdict, dataclass
from functools import partial
from typing import Any

import numpy as np
import numpy.typing as npt

from pr_lens.eval.corpus import RepoCorpus, Serialisation, all_documents, gold_texts, load_repo
from pr_lens.eval.pairs import QUERY_VARIANTS, ReviewPair, query_text
from pr_lens.eval.recall import (
    DENSE_INDEX,
    FUSION_DEPTH,
    KS,
    RERANK_DEPTH,
    DocumentTable,
    assert_no_leakage,
    gold_ranks,
    language,
    length_buckets,
    post_filter,
    recall_at,
    rerank_plateau,
)
from pr_lens.eval.report import render
from pr_lens.eval.split import require_tune, tune_repos
from pr_lens.eval.store import Store, build_store
from pr_lens.eval.vectors import corpus_part, document_matrix, load_rows, query_id, query_part
from pr_lens.jobs.pairs import load_pairs
from pr_lens.jobs.plan import EMBED_PARTS, PAIRS_VERSION, QUERY_PARTS, RERANK_SAMPLE
from pr_lens.logging import configure
from pr_lens.retrieval.bm25 import BM25Index
from pr_lens.retrieval.embed import MODELS, SentenceEncoder
from pr_lens.retrieval.rerank import RERANKER_NAME, RERANKER_REVISION, CrossEncoderReranker
from pr_lens.retrieval.search import FlatIndex, Ids, Mask, reciprocal_rank_fusion, top_k

logger = logging.getLogger(__name__)

SERIALISATIONS: tuple[Serialisation, ...] = ("with_hunk", "body_only")
FILTERS = ("repo AND kind=review_comment", "repo AND language")
LATENCY_SAMPLE = 200
ENCODER_LATENCY_SAMPLE = 30


def results_path(version: str, name: str) -> str:
    return f"eval/results/{version}/{name}"


@dataclass(frozen=True, slots=True)
class Loaded:
    pairs: list[ReviewPair]
    manifest: dict[str, Any]
    corpora: list[RepoCorpus]
    table: DocumentTable
    gold: npt.NDArray[np.int64]


def load(store: Store, version: str) -> Loaded:
    pairs, manifest = load_pairs(store, version)
    # The harness's own repo list. It passes through the split check here, so a held-out
    # repo cannot be indexed even by editing the list above.
    repos = require_tune(tune_repos())
    corpora = [load_repo(store, repo) for repo in repos]
    table = DocumentTable.build(all_documents(corpora))
    logger.info(
        "%s pairs; %s documents across %s repos: %s",
        len(pairs),
        len(table.documents),
        len(corpora),
        dict(Counter(d.kind for d in table.documents)),
    )
    return Loaded(pairs, manifest, corpora, table, table.gold_rows(pairs))


def percentiles(samples: Sequence[float]) -> dict[str, float]:
    if not samples:
        return {"p50_ms": 0.0, "p95_ms": 0.0, "n": 0}
    ordered = sorted(samples)
    p95 = ordered[min(len(ordered) - 1, round(0.95 * (len(ordered) - 1)))]
    return {"p50_ms": 1000 * statistics.median(ordered), "p95_ms": 1000 * p95, "n": len(ordered)}


def time_each[T](call: Callable[[T], object], items: Iterable[T]) -> list[float]:
    """Wall time of one call per item, each measured alone, as serving would pay it."""
    samples = []
    for item in items:
        started = time.perf_counter()
        call(item)
        samples.append(time.perf_counter() - started)
    return samples


def rerank_sample(pairs: Sequence[ReviewPair]) -> list[int]:
    """A fixed pseudo-random sample, by a hash of the comment id, the same on every run."""
    ranked = sorted(
        range(len(pairs)),
        key=lambda i: hashlib.sha256(f"rerank|{pairs[i].comment_id}".encode()).hexdigest(),
    )
    return sorted(ranked[:RERANK_SAMPLE])


def first_stage(store: Store, version: str) -> dict[str, Any]:
    data = load(store, version)
    documents = data.table.documents
    pairs = data.pairs

    assert_no_leakage(pairs, gold_texts(documents, "body_only"))
    logger.info("leakage check passed: no body-only gold contains its query")

    queries = {v: [query_text(p, v) for p in pairs] for v in QUERY_VARIANTS}
    bm25 = {s: BM25Index([d.serialised(s) for d in documents]) for s in SERIALISATIONS}
    bm25_top = {
        (s, v): [top_k(bm25[s].scores(q), FUSION_DEPTH) for q in queries[v]]
        for s in SERIALISATIONS
        for v in QUERY_VARIANTS
    }
    logger.info("BM25 built and queried")

    rows: list[dict[str, Any]] = []
    for s in SERIALISATIONS:
        for v in QUERY_VARIANTS:
            ranks = gold_ranks(bm25_top[(s, v)], data.gold)
            rows.append(_row("bm25", "bm25", s, v, ranks))

    truncation: dict[str, Any] = {}
    buckets: dict[str, Any] = {}
    filters: dict[str, Any] = {}
    candidates: dict[tuple[str, str], list[Ids]] = {}
    latency: dict[str, Any] = {}
    masks = _filter_masks(data.table, pairs)

    for key, model in MODELS.items():
        doc_rows = load_rows(
            store,
            [
                corpus_part(c, model, p, EMBED_PARTS[key])
                for c in data.corpora
                for p in range(EMBED_PARTS[key])
            ],
        )
        query_rows = load_rows(
            store,
            [
                query_part(pairs, str(data.manifest["sha256"]), version, model, p, QUERY_PARTS[key])
                for p in range(QUERY_PARTS[key])
            ],
        )
        vectors = {v: query_rows.take([query_id(p, v) for p in pairs]) for v in QUERY_VARIANTS}

        with_hunk_tokens = document_matrix(doc_rows, documents, body_only=False)[1]
        truncation[key] = {
            "window": model.window,
            "native_tokens": model.native_tokens,
            "units": _overflow(with_hunk_tokens, model.window, data.table.kind),
            "queries": {v: _overflow(vectors[v][1], model.window, None) for v in QUERY_VARIANTS},
        }

        for s in SERIALISATIONS:
            matrix, _ = document_matrix(doc_rows, documents, body_only=s == "body_only")
            index = DENSE_INDEX(matrix)
            for v in QUERY_VARIANTS:
                dense_top = index.search(vectors[v][0], FUSION_DEPTH)
                fused = [
                    np.asarray(reciprocal_rank_fusion([d, b])[:FUSION_DEPTH], dtype=np.int64)
                    for d, b in zip(dense_top, bm25_top[(s, v)], strict=True)
                ]
                dense_ranks = gold_ranks(dense_top, data.gold)
                rows.append(_row(key, "dense", s, v, dense_ranks))
                rows.append(_row(key, "rrf", s, v, gold_ranks(fused, data.gold)))
                if s == "body_only":
                    candidates[(key, v)] = fused
                    buckets.setdefault(key, {})[v] = {
                        "dense": length_buckets(dense_ranks, vectors[v][1], 512),
                        "per_repo_recall_at_10": _per_repo(fused, data.gold, pairs),
                    }
                    filters.setdefault(key, {})[v] = {
                        name: {
                            "selectivity": float(np.mean([m.mean() for m in masks[name]])),
                            "rows": [
                                asdict(r)
                                for r in post_filter(index, vectors[v][0], data.gold, masks[name])
                            ],
                        }
                        for name in FILTERS
                    }
            if s == "body_only":
                latency[f"exact dense search, {key}, {index.size} rows"] = percentiles(
                    time_each(partial(_search_one, index), vectors["reviewed"][0][:LATENCY_SAMPLE])
                )
        logger.info("first stage done for %s", key)

    body_bm25 = bm25["body_only"]
    latency["BM25 score and top 100"] = percentiles(
        time_each(partial(_bm25_one, body_bm25), queries["reviewed"][:LATENCY_SAMPLE])
    )
    latency.update(_encoder_latency(queries["reviewed"][:ENCODER_LATENCY_SAMPLE]))

    # The reranker can only reorder what the first stage found, so it gets the candidates of
    # whichever model's fused list holds the gold most often near the rerank depth.
    rerank_model = max(
        MODELS,
        key=lambda k: np.mean(
            [
                r["recall"]["20"]
                for r in rows
                if r["model"] == k and r["retriever"] == "rrf" and r["serialisation"] == "body_only"
            ]
        ),
    )
    sampled = rerank_sample(pairs)
    items: list[dict[str, Any]] = [
        {
            "pair": i,
            "variant": v,
            "candidates": [
                documents[row].unit_id for row in candidates[(rerank_model, v)][i][:RERANK_DEPTH]
            ],
            "gold": pairs[i].gold_unit_id,
        }
        for v in QUERY_VARIANTS
        for i in sampled
    ]

    result: dict[str, Any] = {
        "version": version,
        "pairs_sha256": data.manifest["sha256"],
        "query_set": data.manifest,
        "corpus": {
            "documents": len(documents),
            "repos": len(data.corpora),
            "by_kind": dict(Counter(d.kind for d in documents)),
        },
        "models": {k: asdict(m) for k, m in MODELS.items()},
        "reranker": {"name": RERANKER_NAME, "revision": RERANKER_REVISION},
        "first_stage": rows,
        "truncation": truncation,
        "buckets": buckets,
        "post_filter": filters,
        "latency": latency,
        "machine": {"platform": platform.platform(), "cpus": os.cpu_count()},
        "neon": asyncio.run(_neon_latency([item["candidates"] for item in items])),
        "rerank": {"model": rerank_model, "items": items},
    }
    store.write({results_path(version, "first-stage.json"): _json(result)})
    return result


def rerank_fingerprint(first: dict[str, Any]) -> str:
    """What a rerank shard is a function of: the candidate lists, and nothing else.

    Not the whole first-stage file, which carries latency timings that differ on every run.
    Re-running the first stage over the same inputs must not throw away hours of reranking.
    """
    return hashlib.sha256(_json(first["rerank"])).hexdigest()


def rerank(store: Store, version: str, shard: int, shards: int) -> None:
    raw = store.read(results_path(version, "first-stage.json"))
    if raw is None:
        raise FileNotFoundError("run the first-stage step before rerank")
    first = json.loads(raw)
    out_path = results_path(version, f"rerank/shard-{shard:02d}-of-{shards:02d}.json")
    fingerprint = rerank_fingerprint(first)
    existing = store.read(out_path)
    if existing is not None and json.loads(existing).get("candidates_sha256") == fingerprint:
        logger.info("%s already scored for this first stage, nothing to do", out_path)
        return

    data = load(store, version)
    texts = {d.unit_id: d.serialised("body_only") for d in data.table.documents}
    reranker = CrossEncoderReranker(max_tokens=MODELS["gte-modernbert"].window)
    scored = []
    seconds = []
    for item in first["rerank"]["items"][shard::shards]:
        pair = data.pairs[item["pair"]]
        started = time.perf_counter()
        scores = reranker.score(
            query_text(pair, item["variant"]), [texts[u] for u in item["candidates"]]
        )
        seconds.append(time.perf_counter() - started)
        scored.append({"pair": item["pair"], "variant": item["variant"], "scores": scores.tolist()})
    store.write(
        {out_path: _json({"candidates_sha256": fingerprint, "items": scored, "seconds": seconds})}
    )
    logger.info("%s: %s rerank passes, %s", out_path, len(scored), percentiles(seconds))


def table(store: Store, version: str, shards: int) -> str:
    raw = store.read(results_path(version, "first-stage.json"))
    if raw is None:
        raise FileNotFoundError("run the first-stage step first")
    first = json.loads(raw)
    fingerprint = rerank_fingerprint(first)
    scores: dict[tuple[int, str], list[float]] = {}
    seconds: list[float] = []
    for shard in range(shards):
        payload = store.read(
            results_path(version, f"rerank/shard-{shard:02d}-of-{shards:02d}.json")
        )
        if payload is None:
            raise FileNotFoundError(f"rerank shard {shard} of {shards} is missing")
        part = json.loads(payload)
        if part["candidates_sha256"] != fingerprint:
            raise ValueError(
                f"rerank shard {shard} scored other candidates, from an older first stage"
            )
        seconds.extend(part["seconds"])
        for item in part["items"]:
            scores[(item["pair"], item["variant"])] = item["scores"]

    # Unit ids as small integers, which is all the plateau arithmetic needs.
    number: dict[str, int] = {}

    def rows_of(ids: Sequence[str]) -> Ids:
        return np.asarray([number.setdefault(u, len(number)) for u in ids], dtype=np.int64)

    plateau: dict[str, Any] = {}
    for v in QUERY_VARIANTS:
        items = [i for i in first["rerank"]["items"] if i["variant"] == v]
        plateau[v] = [
            asdict(row)
            for row in rerank_plateau(
                [rows_of(i["candidates"]) for i in items],
                [np.asarray(scores[(i["pair"], v)], dtype=np.float32) for i in items],
                rows_of([i["gold"] for i in items]),
            )
        ]
    first["plateau"] = plateau
    first["latency"][f"rerank top {RERANK_DEPTH}, one query"] = percentiles(seconds)
    # Rendered from the stored form, so the markdown is exactly what the JSON can reproduce.
    stored = _json(first)
    markdown = render(json.loads(stored))
    store.write(
        {
            results_path(version, "recall-table.md"): markdown.encode(),
            results_path(version, "recall-table.json"): stored,
        }
    )
    return markdown


def _row(
    model: str, retriever: str, s: str, v: str, ranks: npt.NDArray[np.int64]
) -> dict[str, Any]:
    return {
        "model": model,
        "retriever": retriever,
        "serialisation": s,
        "variant": v,
        "n": int(ranks.size),
        "recall": {str(k): value for k, value in recall_at(ranks, KS).items()},
    }


def _overflow(
    tokens: npt.NDArray[np.int32], window: int, kinds: npt.NDArray[np.str_] | None
) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "n": int(tokens.size),
        "over_512": float((tokens > 512).mean()) if tokens.size else 0.0,
        "over_window": float((tokens > window).mean()) if tokens.size else 0.0,
        "median_tokens": float(np.median(tokens)) if tokens.size else 0.0,
    }
    if kinds is not None:
        summary["over_512_by_kind"] = {
            str(kind): float((tokens[kinds == kind] > 512).mean()) for kind in np.unique(kinds)
        }
    return summary


def _filter_masks(table: DocumentTable, pairs: Sequence[ReviewPair]) -> dict[str, list[Mask]]:
    """One mask per query, shared between queries that ask the same filtered question."""
    cache: dict[tuple[str, str, str], Mask] = {}

    def mask(name: str, pair: ReviewPair) -> Mask:
        if name == FILTERS[0]:
            key = (name, pair.repo, "review_comment")
            column = table.kind
        else:
            key = (name, pair.repo, language(pair.path))
            column = table.language
        if key not in cache:
            cache[key] = (table.repo == pair.repo) & (column == key[2])
        return cache[key]

    return {name: [mask(name, p) for p in pairs] for name in FILTERS}


def _per_repo(
    rankings: Sequence[Ids], gold: npt.NDArray[np.int64], pairs: Sequence[ReviewPair]
) -> dict[str, float]:
    ranks = gold_ranks(rankings, gold)
    repos = np.array([p.repo for p in pairs])
    return {repo: recall_at(ranks[repos == repo], (10,))[10] for repo in sorted(set(repos))}


def _search_one(index: FlatIndex, query: npt.NDArray[np.float32]) -> None:
    index.search(query[None], FUSION_DEPTH)


def _bm25_one(index: BM25Index, query: str) -> None:
    top_k(index.scores(query), FUSION_DEPTH)


def _encode_one(encoder: SentenceEncoder, query: str) -> None:
    encoder.encode([query])


def _encoder_latency(queries: Sequence[str]) -> dict[str, Any]:
    latency = {}
    for key, model in MODELS.items():
        encoder = SentenceEncoder(model, batch_size=1)
        encoder.encode(queries[:1])
        latency[f"encode one query, {key}"] = percentiles(
            time_each(partial(_encode_one, encoder), queries)
        )
    return latency


async def _neon_latency(candidate_lists: Sequence[Sequence[str]]) -> dict[str, Any]:
    """The first query of the run, which on Neon free is usually a wake from scale-to-zero,
    and then the warm cost of resolving one candidate list in one round trip."""
    dsn = os.environ.get("DATABASE_URL")
    if not dsn:
        return {"skipped": "DATABASE_URL is not set"}
    from pr_lens.db.connection import connect
    from pr_lens.db.corpus import resolve_units

    started = time.perf_counter()
    conn = await connect(dsn)
    try:
        await resolve_units(conn, candidate_lists[0])
        first = time.perf_counter() - started
        warm = []
        for ids in candidate_lists[1:LATENCY_SAMPLE]:
            t = time.perf_counter()
            await resolve_units(conn, ids)
            warm.append(time.perf_counter() - t)
    finally:
        await conn.close()
    return {"first_query_including_connect_ms": 1000 * first, "resolve_30": percentiles(warm)}


def _json(value: object) -> bytes:
    return json.dumps(value, indent=1, sort_keys=True).encode()


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="pr-lens-recall", description=__doc__)
    parser.add_argument("step", choices=("first-stage", "rerank", "table"))
    parser.add_argument("--version", default=PAIRS_VERSION)
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--shards", type=int, default=1)
    parser.add_argument("--sink", choices=("local", "huggingface"), default="local")
    parser.add_argument("--corpus-dir", default=".corpus")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    configure(os.environ.get("LOG_LEVEL", "INFO"))
    args = parse_args(argv)
    store = build_store(args.sink, args.corpus_dir)
    if args.step == "first-stage":
        first_stage(store, args.version)
    elif args.step == "rerank":
        rerank(store, args.version, args.shard, args.shards)
    else:
        markdown = table(store, args.version, args.shards)
        sys.stdout.write(markdown)
        summary = os.environ.get("GITHUB_STEP_SUMMARY")
        if summary:
            with open(summary, "a", encoding="utf-8") as handle:
                handle.write(markdown)
    return 0


if __name__ == "__main__":
    sys.exit(main())
