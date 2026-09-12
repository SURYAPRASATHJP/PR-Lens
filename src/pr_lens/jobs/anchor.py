"""The public anchor: reproduce a published number, so the harness is known to be sound.

Without it, a corpus recall of 0.45 has two explanations, a weak retriever or a hard
corpus, and no way to choose between them. gte-modernbert-base publishes nDCG@10 of 88.16
on CoIR's CodeSearchNet python task. If this harness, run on that data, lands near 88.16,
then the harness is right and the Phase 2 numbers mean what they say. If it does not, the
harness is broken, and it is far better to find that out here.

Same setup as the MTEB task that produced the published number: dataset revision
4adc7bc4, the test-partition queries, every document in the corpus. The one difference is
stated: a fixed sample of test queries rather than all 14,918, because the long-context
model on a CPU is the cost, and the interval printed beside the result says what that
sample buys. In this task the query is code and the document is its docstring, which is
the same direction as Phase 2's hunk-to-comment retrieval.

The dataset card states no licence (checked 12 Sep 2026). Nothing derived from it is
committed except the aggregate scores.
"""

import argparse
import hashlib
import json
import logging
import os
import sys
from collections import defaultdict
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from pr_lens.eval.recall import interval
from pr_lens.eval.store import Store, build_store
from pr_lens.jobs.plan import ANCHOR_PARTS
from pr_lens.logging import configure
from pr_lens.retrieval.embed import MODELS, SentenceEncoder, embed_part, load_part, part_fingerprint
from pr_lens.retrieval.search import FlatIndex

logger = logging.getLogger(__name__)

DATASET = "CoIR-Retrieval/CodeSearchNet"
DATASET_REVISION = "4adc7bc41202b5c13543c9c886a25f340634dab3"
MODEL = MODELS["gte-modernbert"]
PUBLISHED_NDCG_AT_10 = 0.8816
QUERY_SAMPLE = 1000
K = 10


def download(filename: str) -> str:
    from huggingface_hub import hf_hub_download

    path: str = hf_hub_download(DATASET, filename, repo_type="dataset", revision=DATASET_REVISION)
    return path


def corpus() -> tuple[list[str], list[str]]:
    """Every document, ordered by id, as MTEB builds it: title and text joined."""
    import pyarrow.parquet as pq

    table = pq.read_table(
        download("python-corpus/corpus-00000-of-00001.parquet"), columns=["_id", "title", "text"]
    ).to_pylist()
    table.sort(key=lambda row: str(row["_id"]))
    ids = [str(row["_id"]) for row in table]
    texts = [f"{row['title'] or ''} {row['text'] or ''}".strip() for row in table]
    return ids, texts


def test_queries() -> tuple[dict[str, str], dict[str, dict[str, int]]]:
    import pyarrow.parquet as pq

    qrels: dict[str, dict[str, int]] = defaultdict(dict)
    for row in pq.read_table(download("python-qrels/test-00000-of-00001.parquet")).to_pylist():
        qrels[str(row["query-id"])][str(row["corpus-id"])] = int(row["score"])
    queries = {
        str(row["_id"]): str(row["text"])
        for row in pq.read_table(
            download("python-queries/queries-00000-of-00001.parquet"),
            columns=["_id", "text", "partition"],
        ).to_pylist()
        if row["partition"] == "test" and str(row["_id"]) in qrels
    }
    return queries, dict(qrels)


def sample(query_ids: Sequence[str], size: int) -> list[str]:
    ranked = sorted(query_ids, key=lambda q: hashlib.sha256(f"anchor|{q}".encode()).hexdigest())
    return sorted(ranked[:size])


def ndcg_at_k(ranked: Sequence[str], relevant: Mapping[str, int], k: int = K) -> float:
    """Linear gain, as trec_eval's ndcg_cut computes it, which is what MTEB reports."""
    dcg = sum(relevant.get(doc, 0) / np.log2(rank + 2) for rank, doc in enumerate(ranked[:k]))
    ideal = sorted(relevant.values(), reverse=True)[:k]
    idcg = sum(gain / np.log2(rank + 2) for rank, gain in enumerate(ideal))
    return float(dcg / idcg) if idcg else 0.0


def part_path(part: int, parts: int) -> str:
    return f"eval/anchor/{MODEL.key}/corpus-part-{part:02d}-of-{parts:02d}"


def fingerprint(part: int, parts: int) -> str:
    return part_fingerprint(f"{DATASET}@{DATASET_REVISION}|python-corpus", MODEL, part, parts)


def embed(store: Store, part: int, parts: int) -> None:
    ids, texts = corpus()
    embed_part(
        store,
        part_path(part, parts),
        fingerprint(part, parts),
        ids[part::parts],
        texts[part::parts],
        SentenceEncoder(MODEL),
    )


def score(store: Store, parts: int) -> dict[str, Any]:
    stored = [load_part(store, part_path(p, parts), fingerprint(p, parts)) for p in range(parts)]
    doc_ids = [i for part in stored for i in part.ids]
    index = FlatIndex(np.concatenate([part.vectors for part in stored]))

    queries, qrels = test_queries()
    chosen = sample(sorted(queries), QUERY_SAMPLE)
    vectors = SentenceEncoder(MODEL).encode([queries[q] for q in chosen])
    ranked = index.search(vectors, K)

    ndcg = [
        ndcg_at_k([doc_ids[r] for r in rows], qrels[q])
        for q, rows in zip(chosen, ranked, strict=True)
    ]
    hits = [
        any(doc_ids[r] in qrels[q] for r in rows) for q, rows in zip(chosen, ranked, strict=True)
    ]
    mean = float(np.mean(ndcg))
    spread = 1.96 * float(np.std(ndcg, ddof=1)) / np.sqrt(len(ndcg))
    result: dict[str, Any] = {
        "dataset": f"{DATASET}@{DATASET_REVISION[:8]}",
        "task": "CodeSearchNet python, code to docstring",
        "model": f"{MODEL.name}@{MODEL.revision[:10]}",
        "window": MODEL.window,
        "corpus_documents": len(doc_ids),
        "test_queries": len(queries),
        "queries_scored": len(chosen),
        "ndcg_at_10": mean,
        "ndcg_at_10_interval": spread,
        "recall_at_10": float(np.mean(hits)),
        "recall_at_10_interval": interval(float(np.mean(hits)), len(hits)),
        "published_ndcg_at_10": PUBLISHED_NDCG_AT_10,
        "published_source": "Alibaba-NLP/gte-modernbert-base model card, CoIR table",
    }
    store.write({"eval/anchor/result.json": json.dumps(result, indent=1).encode()})
    return result


def render(result: Mapping[str, Any]) -> str:
    return "\n".join(
        [
            "## Public anchor, CoIR CodeSearchNet python",
            "",
            f"{result['model']}, window {result['window']} tokens, exact search over all "
            f"{result['corpus_documents']:,} documents, {result['queries_scored']} of "
            f"{result['test_queries']:,} test queries sampled by a hash of the query id.",
            "",
            "| | nDCG@10 | recall@10 |",
            "|---|---|---|",
            f"| measured here | {result['ndcg_at_10']:.4f} ± {result['ndcg_at_10_interval']:.4f} "
            f"| {result['recall_at_10']:.4f} ± {result['recall_at_10_interval']:.4f} |",
            f"| published, quoted not measured | {result['published_ndcg_at_10']:.4f} | |",
            "",
            f"Published figure from the {result['published_source']}.",
            "",
        ]
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="pr-lens-anchor", description=__doc__)
    parser.add_argument("step", choices=("embed", "score"))
    parser.add_argument("--part", type=int, default=0)
    parser.add_argument("--parts", type=int, default=ANCHOR_PARTS)
    parser.add_argument("--sink", choices=("local", "huggingface"), default="local")
    parser.add_argument("--corpus-dir", default=".corpus")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    configure(os.environ.get("LOG_LEVEL", "INFO"))
    args = parse_args(argv)
    store = build_store(args.sink, args.corpus_dir)
    if args.step == "embed":
        embed(store, args.part, args.parts)
        return 0
    markdown = render(score(store, args.parts))
    sys.stdout.write(markdown)
    summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary:
        with open(summary, "a", encoding="utf-8") as handle:
            handle.write(markdown)
    return 0


if __name__ == "__main__":
    sys.exit(main())
