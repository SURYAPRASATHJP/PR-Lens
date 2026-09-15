"""Review one pull request live: draft, gate, filter, and post what survives.

Run by review.yml after record_delivery has taken the delivery row, with an installation
token minted for the one repository under review and scoped to pull requests write and
contents read. The same pipeline replay runs, with two differences: lines that already
carry a comment are left alone, and what is kept is posted, each comment only after its
posted_comments row is taken.

Silence is recorded like everything else. A pull request that is a draft or opened by a
bot gets no review. A repository with no corpus yet is reviewed without past comments.

The testbed is the one place a branch name means something. A pull request there whose
head branch is seed/<owner>__<name>/<number>/head is a historical pull request rebuilt for
a live run, and it retrieves from the repository it was copied out of, stopping below the
original number exactly as replay does. Every other repository's branch names are only
names, since anyone opening a pull request chooses them.
"""

import asyncio
import logging
import os
import re
import sys
import time

import httpx

from pr_lens.corpus.writer import repo_slug
from pr_lens.db import reviews
from pr_lens.db.connection import connect
from pr_lens.eval.split import HoldoutViolation
from pr_lens.eval.store import build_store
from pr_lens.github.cache import HttpCache
from pr_lens.github.client import GitHubClient
from pr_lens.github.mining import DEFAULT_CACHE_DIR
from pr_lens.jobs.record_delivery import parse_payload
from pr_lens.logging import configure
from pr_lens.retrieval.embed import MODELS, SentenceEncoder
from pr_lens.review.context import fetch_changes, fetch_pull, fetch_windows
from pr_lens.review.pipeline import MAX_CANDIDATE_HUNKS, review
from pr_lens.review.post import post
from pr_lens.review.provider import from_env
from pr_lens.review.retrieve import CommentIndex, load_index

logger = logging.getLogger(__name__)

TESTBED = "suryaprasathjp/pr-lens-testbed"
_SEED = re.compile(r"^seed/(?P<owner>[\w.-]+)__(?P<name>[\w.-]+)/(?P<number>[1-9]\d*)/head$")


def seed_source(repo: str, head_ref: str) -> tuple[str, int] | None:
    """The repository and pull request a seeded testbed pull request was copied from."""
    if repo.lower() != TESTBED:
        return None
    match = _SEED.match(head_ref)
    if match is None:
        return None
    return f"{match['owner']}/{match['name']}", int(match["number"])


def skip_reason(draft: bool, author: str) -> str | None:
    if draft:
        return "a draft pull request, reviewed when it is marked ready"
    if author.endswith("[bot]"):
        return "opened by a bot"
    return None


async def commented_lines(
    client: GitHubClient, repo: str, number: int
) -> frozenset[tuple[str, int]]:
    """Every (path, line) that already carries a review comment, a human's or an earlier run's."""
    lines = set()
    async for comment in client.paginate(f"/repos/{repo}/pulls/{number}/comments"):
        if comment.get("path") and isinstance(comment.get("line"), int):
            lines.add((comment["path"], comment["line"]))
    return frozenset(lines)


def load_comment_index(repo: str) -> CommentIndex | None:
    """The repository's past review comments, or None when there is no corpus to read."""
    store = build_store("huggingface", "")
    if not store.read_manifest(repo_slug(repo)):
        logger.info("%s has no corpus yet; reviewing without past comments", repo)
        return None
    try:
        return load_index(store, repo, SentenceEncoder(MODELS["gte-modernbert"]))
    except HoldoutViolation:
        logger.info("%s is in the holdout; reviewing without past comments", repo)
        return None


async def run(raw_payload: str, dsn: str, token: str) -> str:
    delivery = parse_payload(raw_payload)
    repo, number = delivery.repo_full_name, delivery.pr_number
    if repo is None or number is None:
        return "Not a pull request event."
    timings: dict[str, float] = {}
    started = time.monotonic()
    async with httpx.AsyncClient() as http:
        client = GitHubClient(http, token, HttpCache(DEFAULT_CACHE_DIR))
        pull = await fetch_pull(client, repo, number)
        reason = skip_reason(pull.draft, pull.author)
        if reason:
            return f"{repo}#{number} not reviewed: {reason}."
        seed = seed_source(repo, pull.head_ref)
        conn = await connect(dsn)
        try:
            run_id = await reviews.claim(
                conn,
                mode="live",
                repo=repo,
                pr_number=number,
                head_sha=pull.head_sha,
                delivery_id=delivery.delivery_id,
                source_repo=seed[0] if seed else None,
                source_pr=seed[1] if seed else None,
            )
            if run_id is None:
                return f"{repo}#{number}: delivery {delivery.delivery_id} already reviewed."
            hunks, skipped = await fetch_changes(client, repo, number)
            windows = await fetch_windows(client, repo, pull.head_sha, hunks[:MAX_CANDIDATE_HUNKS])
            existing = await commented_lines(client, repo, number)
            timings["fetch"] = time.monotonic() - started

            started = time.monotonic()
            index = load_comment_index(seed[0] if seed else repo)
            timings["index"] = time.monotonic() - started

            started = time.monotonic()
            result = await review(
                pull,
                hunks,
                skipped,
                windows,
                from_env(http),
                index=index,
                past_before=seed[1] if seed else number,
                existing=existing,
            )
            timings["review"] = time.monotonic() - started
            await reviews.finish(conn, run_id, result, timings, head_sha=pull.head_sha)

            started = time.monotonic()
            posted = await post(
                conn,
                http,
                token,
                repo=repo,
                pr_number=number,
                head_sha=pull.head_sha,
                run_id=run_id,
                items=result.kept,
            )
            timings["post"] = time.monotonic() - started
        finally:
            await conn.close()

    outcome = result.no_comment.value if result.no_comment else "commented"
    lines = [
        f"## {repo}#{number}",
        "",
        f"Outcome: {outcome}. {len(result.drafts)} drafted, {len(result.kept)} kept, "
        f"{sum(1 for p in posted if p.comment_id)} posted.",
    ]
    if seed:
        lines.append(f"Seeded from {seed[0]}#{seed[1]}; retrieval stopped below it.")
    for posting in posted:
        if posting.comment_id:
            lines.append(f"- {posting.path}:{posting.line} posted as {posting.comment_id}")
        else:
            lines.append(f"- {posting.path}:{posting.line} not posted {posting.error}".rstrip())
    lines.append("Timings: " + ", ".join(f"{k} {v:.1f}s" for k, v in timings.items()))
    return "\n".join(lines) + "\n"


def main() -> int:
    configure(os.environ.get("LOG_LEVEL", "INFO"))
    dsn = os.environ.get("DATABASE_URL")
    token = os.environ.get("GH_INSTALLATION_TOKEN")
    raw = os.environ.get("CLIENT_PAYLOAD")
    if not dsn or not token or not raw:
        logger.error("DATABASE_URL, GH_INSTALLATION_TOKEN and CLIENT_PAYLOAD are all required")
        return 1
    markdown = asyncio.run(run(raw, dsn, token))
    sys.stdout.write(markdown)
    summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary:
        with open(summary, "a", encoding="utf-8") as handle:
            handle.write(markdown)
    return 0


if __name__ == "__main__":
    sys.exit(main())
