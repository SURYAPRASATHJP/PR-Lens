"""The posting ledger. The two 12 Sep wishes that were still wishes, the per-PR cap and one
comment per line under concurrency, are enforced here by constraints and proved here."""

import asyncio
import json
import re
from collections.abc import AsyncIterator
from pathlib import Path

import asyncpg
import httpx
import pytest
import respx

from pr_lens.db.connection import connect
from pr_lens.db.migrate import MIGRATIONS_DIR, migrate
from pr_lens.github.writes import DISCLOSURE
from pr_lens.review.pipeline import MAX_COMMENTS_PER_PR, DraftItem
from pr_lens.review.post import Posted, post

from .conftest import database_url, requires_postgres

COMMENTS = "https://api.github.com/repos/o/r/pulls/7/comments"


def item(line: int, body: str = "Popping keys[0] on an empty dict raises KeyError.") -> DraftItem:
    return DraftItem(
        path="pkg/cache.py",
        line=line,
        body=body,
        evidence="",
        critique="",
        specific=True,
        non_obvious=True,
        grounded=True,
    )


def test_the_ledger_cap_is_the_pipeline_cap() -> None:
    sql = (MIGRATIONS_DIR / "0004_posted_comments.sql").read_text(encoding="utf-8")
    match = re.search(r"check \(slot between 1 and (\d+)\)", sql)
    assert match is not None
    assert int(match[1]) == MAX_COMMENTS_PER_PR


@pytest.fixture
async def conn() -> AsyncIterator[asyncpg.Connection]:
    dsn = database_url()
    assert dsn is not None
    await migrate(dsn)
    connection = await connect(dsn)
    await connection.execute("truncate posted_comments")
    yield connection
    await connection.close()


class Created:
    """GitHub accepting each comment with a fresh id."""

    def __init__(self) -> None:
        self.count = 0

    def __call__(self, request: httpx.Request) -> httpx.Response:
        del request
        self.count += 1
        return httpx.Response(201, json={"id": 1000 + self.count})


async def run_post(conn: asyncpg.Connection, items: list[DraftItem]) -> list[Posted]:
    async with httpx.AsyncClient() as client:
        return list(
            await post(
                conn,
                client,
                "installation-token",
                repo="o/r",
                pr_number=7,
                head_sha="h" * 40,
                run_id=None,
                items=items,
            )
        )


@requires_postgres
@respx.mock
async def test_the_cap_holds_however_many_comments_are_offered(conn: asyncpg.Connection) -> None:
    route = respx.post(COMMENTS).mock(side_effect=Created())
    posted = await run_post(conn, [item(line) for line in range(1, MAX_COMMENTS_PER_PR + 2)])
    assert route.call_count == MAX_COMMENTS_PER_PR
    assert [p.slot for p in posted] == [*range(1, MAX_COMMENTS_PER_PR + 1), None]
    assert await conn.fetchval("select count(*) from posted_comments") == MAX_COMMENTS_PER_PR
    body = json.loads(route.calls[0].request.content)["body"]
    assert body.endswith(DISCLOSURE)


@requires_postgres
@respx.mock
async def test_two_runs_posting_one_comment_at_once_post_it_once(
    conn: asyncpg.Connection,
) -> None:
    """Production check 1, the failure the project is pitched against: two runs on one pull
    request, both sure the comment is worth leaving."""
    route = respx.post(COMMENTS).mock(side_effect=Created())
    dsn = database_url()
    assert dsn is not None
    first, second = await connect(dsn), await connect(dsn)
    try:
        await asyncio.gather(run_post(first, [item(4)]), run_post(second, [item(4)]))
    finally:
        await first.close()
        await second.close()
    assert route.call_count == 1
    assert await conn.fetchval("select count(*) from posted_comments") == 1


@requires_postgres
@respx.mock
async def test_a_line_an_earlier_run_commented_on_is_never_commented_on_again(
    conn: asyncpg.Connection,
) -> None:
    route = respx.post(COMMENTS).mock(side_effect=Created())
    await run_post(conn, [item(4)])
    later = await run_post(conn, [item(4, "A different wording of the same point."), item(5)])
    assert [p.slot for p in later] == [None, 2]
    assert route.call_count == 2


@requires_postgres
@respx.mock
async def test_a_failed_post_keeps_its_slot_and_is_not_retried(
    conn: asyncpg.Connection,
) -> None:
    route = respx.post(COMMENTS).mock(return_value=httpx.Response(422, text="line not in diff"))
    [failed] = await run_post(conn, [item(4)])
    assert route.call_count == 1
    assert failed.error.startswith("422")
    row = await conn.fetchrow("select status, error from posted_comments")
    assert row is not None
    assert row["status"] == "failed"
    again = await run_post(conn, [item(4)])
    assert route.call_count == 1
    assert again[0].slot is None


def test_nothing_here_posts_without_the_ledger() -> None:
    """post.py is the one caller of the comment route, and it takes the row first."""
    source = Path(__file__).resolve().parents[1] / "src" / "pr_lens" / "review" / "post.py"
    text = source.read_text(encoding="utf-8")
    assert text.index("comments.take_slot(") < text.index("writes.send(")
