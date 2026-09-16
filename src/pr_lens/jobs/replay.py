"""Replay the pipeline over real merged pull requests, drafting and never posting.

The App is installed on one testbed, which is too few pull requests to judge a reviewer
by. So the keep-or-kill loop reads drafts made over pull requests the 18 tune repos have
already merged, chosen from the Phase 2 pairs so that each one had a human line comment:
the user reads the draft beside what the human actually said. The holdout never appears.
Every pair is tune by construction, require_tune checks it anyway, and retrieval refuses
the holdout on its own account.

Replay never posts, and the enforcer is the credential rather than this code: replay.yml
holds no App key and no write permission, which test_workflows checks. The human comments
go into the run's reference column for the verdicts file and nowhere near the pipeline.

A batch is a name. Each pull request is claimed in review_runs before any tokens are
spent, so re-running a batch after a failure drafts only what is left, and pull requests
any earlier batch drafted are not chosen again.
"""

import argparse
import asyncio
import hashlib
import logging
import os
import sys
import time
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass

import httpx

from pr_lens.db import reviews
from pr_lens.db.connection import connect
from pr_lens.eval.pairs import ReviewPair
from pr_lens.eval.split import require_tune
from pr_lens.eval.store import Store, build_store
from pr_lens.github.cache import HttpCache
from pr_lens.github.client import GitHubClient, GitHubError
from pr_lens.github.mining import DEFAULT_CACHE_DIR, mining_token
from pr_lens.jobs.pairs import load_pairs
from pr_lens.jobs.plan import PAIRS_VERSION
from pr_lens.logging import configure
from pr_lens.retrieval.embed import MODELS, Encoder, SentenceEncoder
from pr_lens.review.context import fetch_changes, fetch_pull, fetch_windows
from pr_lens.review.pipeline import MAX_CANDIDATE_HUNKS, NoComment, Review, review
from pr_lens.review.provider import Inference, from_env
from pr_lens.review.retrieve import CommentIndex, load_index
from pr_lens.review.verify import verify

logger = logging.getLogger(__name__)

# About 8,500 tokens a pull request against Groq's 200,000 a day for gpt-oss-120b, with
# room left for the provider check and a live review or two.
DEFAULT_PULLS = 15

# The human's comment is for the reader of the verdicts file, not a document to archive.
MAX_REFERENCE_CHARS = 800


@dataclass(frozen=True, slots=True)
class Chosen:
    repo: str
    number: int
    pairs: tuple[ReviewPair, ...]

    @property
    def reference(self) -> list[dict[str, object]]:
        return [
            {
                "path": pair.path,
                "author": pair.author,
                "body": pair.body[:MAX_REFERENCE_CHARS],
                "url": pair.html_url,
            }
            for pair in self.pairs
        ]


def choose(
    pairs: Sequence[ReviewPair], count: int, batch: str, exclude: set[tuple[str, int]]
) -> list[Chosen]:
    """Pull requests spread across repos, in an order the batch name fixes.

    Round robin over repos, so a batch of fifteen is not fifteen pulls of the one repo with
    the most review traffic, and a shuffle seeded by the batch name, so the same batch
    always means the same pull requests and a new batch means different ones.
    """
    require_tune({pair.repo for pair in pairs})
    grouped: dict[tuple[str, int], list[ReviewPair]] = defaultdict(list)
    for pair in pairs:
        grouped[(pair.repo, pair.pull_request)].append(pair)

    def shuffled(key: tuple[str, int]) -> str:
        return hashlib.sha256(f"{batch}|{key[0]}|{key[1]}".encode()).hexdigest()

    by_repo: dict[str, list[tuple[str, int]]] = defaultdict(list)
    for key in sorted(grouped, key=shuffled):
        if key not in exclude:
            by_repo[key[0]].append(key)
    queues = [by_repo[repo] for repo in sorted(by_repo, key=lambda r: shuffled((r, 0)))]

    chosen: list[Chosen] = []
    while len(chosen) < count and any(queues):
        for queue in queues:
            if queue and len(chosen) < count:
                repo, number = queue.pop(0)
                chosen.append(Chosen(repo, number, tuple(grouped[(repo, number)])))
    return chosen


async def replay_one(
    client: GitHubClient,
    chosen: Chosen,
    index: CommentIndex,
    inference: Inference,
    sandbox: bool = False,
) -> tuple[Review, str, dict[str, float]]:
    """One pull request through the same pipeline live review uses. Returns the review, the
    head sha it was made at, and the seconds each stage took."""
    timings: dict[str, float] = {}
    started = time.monotonic()
    try:
        pull = await fetch_pull(client, chosen.repo, chosen.number)
        hunks, skipped = await fetch_changes(client, chosen.repo, chosen.number)
        windows = await fetch_windows(
            client, chosen.repo, pull.head_sha, hunks[:MAX_CANDIDATE_HUNKS]
        )
    except GitHubError as error:
        logger.warning("%s#%s could not be read: %s", chosen.repo, chosen.number, error)
        return Review(NoComment.FETCH_FAILED, detail=str(error)[:300]), "", timings
    timings["fetch"] = time.monotonic() - started

    verification = None
    if sandbox:
        started = time.monotonic()
        verification = await verify(chosen.repo, pull.base_sha, pull.head_sha, hunks)
        timings["verify"] = time.monotonic() - started

    started = time.monotonic()
    result = await review(
        pull,
        hunks,
        skipped,
        windows,
        inference,
        index=index,
        past_before=chosen.number,
        verification=verification,
    )
    timings["review"] = time.monotonic() - started
    return result, pull.head_sha, timings


def spent_the_day(result: Review) -> bool:
    """Rate limited before a single draft existed, which at this tier means a daily cap."""
    return result.no_comment is NoComment.RATE_LIMITED and not result.drafts


def render(batch: str, rows: Sequence[tuple[Chosen, Review]]) -> str:
    lines = [
        f"## Replay batch `{batch}`",
        "",
        "| pull request | outcome | drafts | kept | tokens | hunks shown | past comments |",
        "|---|---|---|---|---|---|---|",
    ]
    for chosen, result in rows:
        tokens = sum(call.prompt_tokens + call.completion_tokens for call in result.calls)
        outcome = result.no_comment.value if result.no_comment else "commented"
        lines.append(
            f"| {chosen.repo}#{chosen.number} | {outcome} | {len(result.drafts)} | "
            f"{len(result.kept)} | {tokens} | {len(result.shown)} | {result.past_shown} |"
        )
    kept = sum(len(result.kept) for _, result in rows)
    silent = sum(1 for _, result in rows if result.no_comment)
    lines += ["", f"{len(rows)} pull requests, {kept} comments kept, {silent} silent."]
    return "\n".join(lines) + "\n"


async def run(
    store: Store, batch: str, count: int, dsn: str, encoder: Encoder, sandbox: bool = False
) -> str:
    pairs, _ = load_pairs(store, PAIRS_VERSION)
    conn = await connect(dsn)
    rows: list[tuple[Chosen, Review]] = []
    indexes: dict[str, CommentIndex] = {}
    try:
        chosen = choose(pairs, count, batch, await reviews.replayed_elsewhere(conn, batch))
        logger.info("batch %s: %s pull requests chosen", batch, len(chosen))
        async with httpx.AsyncClient() as http:
            client = GitHubClient(http, mining_token() or "", HttpCache(DEFAULT_CACHE_DIR))
            await client.assert_mining_budget()
            inference = from_env(http)
            for pick in chosen:
                # head_sha is not known until the pull is read, and the claim has to come
                # first, so the row is claimed with a placeholder and finished with the truth.
                run_id = await reviews.claim(
                    conn,
                    mode="replay",
                    repo=pick.repo,
                    pr_number=pick.number,
                    head_sha="",
                    batch=batch,
                )
                if run_id is None:
                    logger.info("%s#%s already claimed in %s", pick.repo, pick.number, batch)
                    continue
                if pick.repo not in indexes:
                    started = time.monotonic()
                    indexes[pick.repo] = load_index(store, pick.repo, encoder)
                    logger.info(
                        "%s index: %s comments in %.1fs",
                        pick.repo,
                        indexes[pick.repo].size,
                        time.monotonic() - started,
                    )
                result, head_sha, timings = await replay_one(
                    client, pick, indexes[pick.repo], inference, sandbox
                )
                if spent_the_day(result):
                    # Every pull request after this one would come back the same way and
                    # be recorded as drafted when it never was. Give the claim back and stop;
                    # re-running the batch later carries on from here.
                    await reviews.release(conn, run_id)
                    logger.warning("providers are spent for now: %s", result.detail)
                    break
                await reviews.finish(
                    conn, run_id, result, timings, pick.reference, head_sha=head_sha
                )
                rows.append((pick, result))
    finally:
        await conn.close()
    return render(batch, rows)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="pr-lens-replay", description=__doc__)
    parser.add_argument("--batch", required=True)
    parser.add_argument("--pulls", type=int, default=DEFAULT_PULLS)
    parser.add_argument("--sink", choices=("local", "huggingface"), default="huggingface")
    parser.add_argument("--corpus-dir", default=".corpus")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    configure(os.environ.get("LOG_LEVEL", "INFO"))
    args = parse_args(argv)
    dsn = os.environ.get("DATABASE_URL")
    if not dsn:
        logger.error("DATABASE_URL is not set")
        return 1
    encoder = SentenceEncoder(MODELS["gte-modernbert"])
    markdown = asyncio.run(
        run(
            build_store(args.sink, args.corpus_dir),
            args.batch,
            args.pulls,
            dsn,
            encoder,
            os.environ.get("PR_LENS_SANDBOX") == "1",
        )
    )
    sys.stdout.write(markdown)
    summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary:
        with open(summary, "a", encoding="utf-8") as handle:
            handle.write(markdown)
    return 0


if __name__ == "__main__":
    sys.exit(main())
