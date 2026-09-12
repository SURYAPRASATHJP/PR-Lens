"""Mine the Phase 2 query set once, and freeze it.

A query set tuned against is worthless, so this is fixed and written down before any
retrieval choice is measured on it. A version, once written, is never overwritten: a
second run with the same version reads the frozen set back and changes nothing. A
different query set needs a new version name, which then shows in every table built on it.

Reads the raw review-comment payloads through the Phase 1 client and cache, and reads the
corpus only to know which comments made it into the snapshot.
"""

import argparse
import asyncio
import json
import logging
import os
import sys
from collections import Counter
from collections.abc import Sequence
from pathlib import Path

import httpx

from pr_lens.eval.corpus import load_repo
from pr_lens.eval.pairs import (
    RepoPairs,
    ReviewPair,
    cap_per_repo,
    decode_pairs,
    digest,
    encode_pairs,
    manifest_path,
    mine_repo_pairs,
    pairs_path,
)
from pr_lens.eval.split import require_tune, tune_repos
from pr_lens.eval.store import Store, build_store
from pr_lens.github.cache import HttpCache
from pr_lens.github.client import GitHubClient
from pr_lens.ingest.mine import MiningLimits
from pr_lens.jobs.ingest import DEFAULT_CACHE_DIR, mining_token
from pr_lens.logging import configure

logger = logging.getLogger(__name__)


class FrozenPairsError(RuntimeError):
    """The frozen pair file does not match the digest its manifest recorded."""


def load_pairs(store: Store, version: str) -> tuple[list[ReviewPair], dict[str, object]]:
    raw_manifest = store.read(manifest_path(version))
    payload = store.read(pairs_path(version))
    if raw_manifest is None or payload is None:
        raise FileNotFoundError(f"query set {version} has not been frozen. Run the pairs job.")
    manifest: dict[str, object] = json.loads(raw_manifest)
    if digest(payload) != manifest["sha256"]:
        raise FrozenPairsError(f"query set {version} does not match its recorded digest")
    return decode_pairs(payload), manifest


def freeze(store: Store, version: str, results: Sequence[RepoPairs]) -> dict[str, object]:
    cap = cap_per_repo(results)
    pairs = [pair for result in results for pair in result.pairs]
    payload = encode_pairs(pairs)
    totals: Counter[str] = Counter()
    for result in results:
        totals.update(result.dropped)
    manifest: dict[str, object] = {
        "version": version,
        "sha256": digest(payload),
        "pairs": len(pairs),
        "per_repo_cap": cap,
        "with_suggestion_block": sum(1 for p in pairs if p.has_suggestion),
        "dropped": dict(sorted(totals.items())),
        "repos": {
            r.repo: {
                "comments_seen": r.seen,
                "pairs": len(r.pairs),
                "dropped": dict(sorted(r.dropped.items())),
            }
            for r in results
        },
    }
    store.write(
        {
            pairs_path(version): payload,
            manifest_path(version): json.dumps(manifest, indent=2, sort_keys=True).encode(),
        }
    )
    return manifest


async def mine_all(
    store: Store, repos: Sequence[str], client: GitHubClient, limits: MiningLimits
) -> list[RepoPairs]:
    results = []
    for repo in repos:
        corpus = await asyncio.to_thread(load_repo, store, repo)
        comment_units = {d.unit_id for d in corpus.documents if d.kind == "review_comment"}
        results.append(await mine_repo_pairs(client, repo, comment_units, limits))
    return results


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="pr-lens-pairs", description=__doc__)
    parser.add_argument("--version", required=True, help="query set name, for example v1")
    parser.add_argument("--repo", action="append", default=[], help="default: every tune repo")
    parser.add_argument("--sink", choices=("local", "huggingface"), default="local")
    parser.add_argument("--corpus-dir", default=".corpus")
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    return parser.parse_args(argv)


async def run(args: argparse.Namespace) -> int:
    repos = require_tune(args.repo or tune_repos())
    store = build_store(args.sink, args.corpus_dir)

    if store.read(manifest_path(args.version)) is not None:
        pairs, manifest = load_pairs(store, args.version)
        logger.info(
            "query set %s is already frozen: %s pairs, sha256 %s. Nothing written.",
            args.version,
            len(pairs),
            manifest["sha256"],
        )
        return 0

    token = mining_token()
    if not token:
        logger.error("no mining token. Set GH_MINING_TOKEN")
        return 1

    async with httpx.AsyncClient(timeout=60.0, follow_redirects=True) as http:
        client = GitHubClient(http, token, HttpCache(args.cache_dir))
        await client.assert_mining_budget()
        results = await mine_all(store, repos, client, MiningLimits())

    manifest = freeze(store, args.version, results)
    logger.info("froze query set %s", args.version)
    logger.info("%s", json.dumps(manifest, indent=2, sort_keys=True))
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    configure(os.environ.get("LOG_LEVEL", "INFO"))
    return asyncio.run(run(parse_args(argv)))


if __name__ == "__main__":
    sys.exit(main())
