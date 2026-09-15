"""Post the kept drafts as line comments, each one only after its ledger row is taken.

The order is the whole design: take the row in posted_comments, then call GitHub. A run
that loses the race for a slot, or finds its line already commented on, has no row and so
posts nothing. The cap and the one-per-line rule are database constraints, so they hold
across runs, across redeliveries and across a cancelled run that had already begun.

Every body goes out through writes.send, whose comment route refuses a body without the
disclosure line or with any key beyond what one right-side line comment needs.
"""

import hashlib
import logging
from collections.abc import Sequence
from dataclasses import dataclass

import asyncpg
import httpx

from pr_lens.db import comments
from pr_lens.github import writes
from pr_lens.review.pipeline import MAX_COMMENTS_PER_PR, DraftItem

logger = logging.getLogger(__name__)

API_ROOT = "https://api.github.com"


@dataclass(frozen=True, slots=True)
class Posted:
    path: str
    line: int
    slot: int | None
    comment_id: int | None = None
    error: str = ""


async def post(
    conn: asyncpg.Connection,
    client: httpx.AsyncClient,
    token: str,
    *,
    repo: str,
    pr_number: int,
    head_sha: str,
    run_id: int | None,
    items: Sequence[DraftItem],
) -> list[Posted]:
    results: list[Posted] = []
    for item in items:
        body = writes.disclosed(item.body)
        slot = await comments.take_slot(
            conn,
            repo=repo,
            pr_number=pr_number,
            path=item.path,
            line=item.line,
            run_id=run_id,
            body_sha256=hashlib.sha256(body.encode()).hexdigest(),
            cap=MAX_COMMENTS_PER_PR,
        )
        if slot is None:
            logger.info("%s:%s not posted: line taken or cap reached", item.path, item.line)
            results.append(Posted(item.path, item.line, None))
            continue
        try:
            response = await writes.send(
                client,
                "POST",
                f"{API_ROOT}/repos/{repo}/pulls/{pr_number}/comments",
                json={
                    "body": body,
                    "commit_id": head_sha,
                    "path": item.path,
                    "line": item.line,
                    "side": "RIGHT",
                },
                headers={
                    "Authorization": f"Bearer {token}",
                    "Accept": "application/vnd.github+json",
                    "X-GitHub-Api-Version": "2022-11-28",
                },
            )
        except httpx.HTTPError as exc:
            await comments.mark_failed(conn, repo, pr_number, slot, str(exc))
            results.append(Posted(item.path, item.line, slot, error=str(exc)))
            continue
        if response.status_code != httpx.codes.CREATED:
            # Not retried. The slot stays spent, because a retry that half succeeded is the
            # duplicate comment this whole table exists to prevent.
            error = f"{response.status_code}: {response.text[:300]}"
            logger.warning("posting %s:%s failed, %s", item.path, item.line, error)
            await comments.mark_failed(conn, repo, pr_number, slot, error)
            results.append(Posted(item.path, item.line, slot, error=error))
            continue
        comment_id = int(response.json()["id"])
        await comments.mark_posted(conn, repo, pr_number, slot, comment_id)
        results.append(Posted(item.path, item.line, slot, comment_id))
    return results
