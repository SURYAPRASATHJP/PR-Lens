"""The mining job.

Runs on an Actions runner, which is free and unlimited on a public repo but kills a job
at six hours. The 27-repo mining set does not fit in one job, so this is built to be run
repeatedly: the HTTP cache carries across runs, a repo already mined replays from disk in
seconds, and --time-budget stops the run cleanly with time left to save the cache rather
than being killed with it half written.

The repo list is not in this repository. It is passed in, so that the workspace notes it
comes from stay out of a public repo.
"""

import argparse
import asyncio
import json
import logging
import os
import sys
import time
from collections.abc import Sequence
from pathlib import Path

import httpx

from pr_lens.corpus.writer import LocalSink, Sink
from pr_lens.db.connection import connect
from pr_lens.github.cache import HttpCache
from pr_lens.github.client import GitHubClient, GitHubError, NotFound
from pr_lens.ingest.mine import MiningLimits
from pr_lens.ingest.pipeline import ingest_repo
from pr_lens.logging import configure

logger = logging.getLogger(__name__)

DEFAULT_CACHE_DIR = Path(".cache/github")
DEFAULT_CORPUS_DIR = Path(".corpus")

# Six hours is the runner's ceiling. Stopping at five leaves room to save the cache.
DEFAULT_TIME_BUDGET_SECONDS = 5 * 60 * 60

DEFAULT_LIMITS = MiningLimits()


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="pr-lens-ingest", description=__doc__)
    parser.add_argument("--repo", action="append", default=[], help="owner/name, repeatable")
    parser.add_argument(
        "--repos-file",
        type=Path,
        help='json: a list of "owner/name", or the mining-set object with a "repos" key',
    )
    parser.add_argument("--sink", choices=("local", "huggingface"), default="local")
    parser.add_argument("--corpus-dir", type=Path, default=DEFAULT_CORPUS_DIR)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--pull-requests", type=int, default=DEFAULT_LIMITS.pull_requests)
    parser.add_argument("--review-comments", type=int, default=DEFAULT_LIMITS.review_comments)
    parser.add_argument("--issues", type=int, default=DEFAULT_LIMITS.issues)
    parser.add_argument("--time-budget", type=int, default=DEFAULT_TIME_BUDGET_SECONDS)
    return parser.parse_args(argv)


def resolve_repos(args: argparse.Namespace) -> list[str]:
    """--repo, then --repos-file, then MINING_REPOS in the environment."""
    if args.repo:
        return list(args.repo)
    if args.repos_file:
        return _repos_from_json(json.loads(args.repos_file.read_text(encoding="utf-8")))
    raw = os.environ.get("MINING_REPOS", "").strip()
    if not raw:
        return []
    if raw.startswith(("[", "{")):
        return _repos_from_json(json.loads(raw))
    return [item.strip() for item in raw.split(",") if item.strip()]


def mining_token() -> str | None:
    """GH_MINING_TOKEN if the mining run has its own credential, else the dispatch PAT.

    Never the workflow's GITHUB_TOKEN: it is capped at 1,000 requests per hour per
    repository, and the client refuses it rather than running at a fifth of the speed and
    looking like a slow network.
    """
    return os.environ.get("GH_MINING_TOKEN") or os.environ.get("GH_DISPATCH_TOKEN")


def build_sink(args: argparse.Namespace) -> Sink:
    if args.sink == "local":
        logger.info("writing the corpus to %s", args.corpus_dir)
        return LocalSink(args.corpus_dir)
    from pr_lens.corpus.huggingface import HuggingFaceSink

    sink = HuggingFaceSink()
    logger.info("writing the corpus to the dataset repo %s", sink.repo_id)
    return sink


async def run(args: argparse.Namespace) -> int:
    repos = resolve_repos(args)
    if not repos:
        logger.error("no repos to mine. Pass --repo, --repos-file, or set MINING_REPOS")
        return 1

    token = mining_token()
    if not token:
        logger.error("no mining token. Set GH_MINING_TOKEN or GH_DISPATCH_TOKEN")
        return 1

    dsn = os.environ.get("DATABASE_URL")
    if not dsn:
        logger.error("DATABASE_URL is not set")
        return 1

    sink = build_sink(args)
    cache = HttpCache(args.cache_dir)
    limits = MiningLimits(
        pull_requests=args.pull_requests,
        review_comments=args.review_comments,
        issues=args.issues,
    )

    started = time.monotonic()
    remaining = list(repos)
    failed: list[str] = []

    async with httpx.AsyncClient(timeout=60.0, follow_redirects=True) as http:
        client = GitHubClient(http, token, cache)
        await client.assert_mining_budget()
        await preflight(client, repos[0])

        conn = await connect(dsn)
        try:
            for repo in repos:
                if time.monotonic() - started > args.time_budget:
                    logger.warning(
                        "time budget of %ss reached with %s repos left: %s. "
                        "Re-run and the cache resumes from here.",
                        args.time_budget,
                        len(remaining),
                        ", ".join(remaining),
                    )
                    break
                try:
                    await ingest_repo(client, conn, sink, repo, limits)
                except (GitHubError, OSError):
                    # One repo going private or being renamed must not end a run that
                    # still has twenty-six others to do.
                    logger.exception("ingest failed for %s, continuing", repo)
                    failed.append(repo)
                remaining.remove(repo)
        finally:
            await conn.close()

    logger.info(
        "cache: %s hits, %s misses. %s repos done, %s failed, %s not reached",
        cache.hits,
        cache.misses,
        len(repos) - len(remaining) - len(failed),
        len(failed),
        len(remaining),
    )
    return 1 if failed else 0


async def preflight(client: GitHubClient, repo: str) -> None:
    """Prove the token can actually read the mining set before spending an hour on it.

    A fine-grained PAT restricted to selected repositories passes the rate-limit check,
    because the limit is a property of the token rather than of what it can reach, and
    then 404s on every repo it does not own. That failure at hour three looks like the
    mining set being wrong. Here it looks like what it is.
    """
    try:
        await client.get_json(f"/repos/{repo}")
    except NotFound as exc:
        raise GitHubError(
            f"the mining token cannot read {repo}. A 404 on a public repo means the "
            "token has no access to it: a fine-grained PAT limited to selected "
            "repositories cannot read repositories owned by anyone else. Use a token "
            "with public repository read access and pass it as GH_MINING_TOKEN."
        ) from exc


def _repos_from_json(payload: object) -> list[str]:
    if isinstance(payload, dict):
        payload = payload.get("repos", [])
    if not isinstance(payload, list):
        raise ValueError("expected a json list of repos, or an object with a repos key")
    repos = []
    for item in payload:
        if isinstance(item, str):
            repos.append(item)
        elif isinstance(item, dict) and isinstance(item.get("repo"), str):
            repos.append(item["repo"])
    return repos


async def main(argv: Sequence[str] | None = None) -> int:
    configure(os.environ.get("LOG_LEVEL", "INFO"))
    return await run(parse_args(argv))


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
