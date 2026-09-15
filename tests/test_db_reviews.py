"""The review ledger, against the postgres service container in ci.yml."""

import asyncio
from collections.abc import AsyncIterator

import asyncpg
import pytest

from pr_lens.db.connection import connect
from pr_lens.db.migrate import migrate
from pr_lens.db.reviews import batch_drafts, claim, finish, record_verdicts
from pr_lens.ingest.diff import parse_patch
from pr_lens.review.pipeline import Call, Draft, DraftItem, Fate, NoComment, Review

from .conftest import database_url, requires_postgres

pytestmark = requires_postgres

HUNK = parse_patch("@@ -1 +1 @@\n-a\n+b\n", "x.py")[0]


@pytest.fixture
async def conn() -> AsyncIterator[asyncpg.Connection]:
    dsn = database_url()
    assert dsn is not None
    await migrate(dsn)
    connection = await connect(dsn)
    await connection.execute("truncate review_runs, drafts, deliveries cascade")
    yield connection
    await connection.close()


def item(line: int, body: str) -> DraftItem:
    return DraftItem(
        path="x.py",
        line=line,
        body=body,
        evidence="e",
        critique="c",
        specific=True,
        non_obvious=True,
        grounded=True,
    )


async def test_a_replayed_pull_request_is_claimed_once_per_batch(conn: asyncpg.Connection) -> None:
    first = await claim(conn, mode="replay", repo="o/r", pr_number=7, head_sha="h", batch="b1")
    again = await claim(conn, mode="replay", repo="o/r", pr_number=7, head_sha="h", batch="b1")
    other = await claim(conn, mode="replay", repo="o/r", pr_number=7, head_sha="h", batch="b2")
    assert first is not None and other is not None
    assert again is None


async def test_two_jobs_racing_for_one_pull_request_leave_one_winner(
    conn: asyncpg.Connection,
) -> None:
    """Production check 1: fired twice at once, one run row, so one set of drafts."""
    dsn = database_url()
    assert dsn is not None
    first, second = await connect(dsn), await connect(dsn)
    try:
        results = await asyncio.gather(
            claim(first, mode="replay", repo="o/r", pr_number=8, head_sha="h", batch="race"),
            claim(second, mode="replay", repo="o/r", pr_number=8, head_sha="h", batch="race"),
        )
    finally:
        await first.close()
        await second.close()
    assert sum(result is not None for result in results) == 1
    assert await conn.fetchval("select count(*) from review_runs where batch = 'race'") == 1


async def test_a_live_run_answers_exactly_one_delivery(conn: asyncpg.Connection) -> None:
    await conn.execute("insert into deliveries (delivery_id, event) values ('d-1', 'pull_request')")
    run = await claim(conn, mode="live", repo="o/r", pr_number=9, head_sha="h", delivery_id="d-1")
    again = await claim(conn, mode="live", repo="o/r", pr_number=9, head_sha="h", delivery_id="d-1")
    assert run is not None and again is None


@pytest.mark.parametrize(
    ("mode", "delivery_id", "batch"),
    [("replay", None, None), ("live", None, None), ("live", None, "b")],
    ids=["replay without a batch", "live without a delivery", "live in a batch"],
)
async def test_a_run_that_is_neither_one_thing_nor_the_other_is_refused(
    conn: asyncpg.Connection, mode: str, delivery_id: str | None, batch: str | None
) -> None:
    with pytest.raises(asyncpg.CheckViolationError):
        await claim(
            conn,
            mode=mode,  # type: ignore[arg-type]
            repo="o/r",
            pr_number=1,
            head_sha="h",
            delivery_id=delivery_id,
            batch=batch,
        )


async def test_a_finished_run_keeps_every_draft_and_its_verdict(conn: asyncpg.Connection) -> None:
    run = await claim(conn, mode="replay", repo="o/r", pr_number=10, head_sha="h", batch="b")
    assert run is not None
    review = Review(
        None,
        plan="check eviction",
        drafts=[
            Draft(item(1, "kept one"), Fate.KEPT, "specific"),
            Draft(item(2, "dropped one"), Fate.SELF_CRITIQUE),
        ],
        calls=[
            Call("draft", "groq", "m", 1000, 300, 2.5, "stop"),
            Call("filter", "groq", "m", 200, 50, 0.5, "stop"),
        ],
        shown=[HUNK],
        past_shown=4,
    )
    await finish(conn, run, review, {"draft": 2.5, "filter": 0.5})

    row = await conn.fetchrow("select * from review_runs where run_id = $1", run)
    assert row is not None
    assert row["no_comment"] is None
    assert (row["prompt_tokens"], row["completion_tokens"]) == (1200, 350)
    assert (row["hunks_shown"], row["past_shown"], row["provider"]) == (1, 4, "groq")

    drafts = await batch_drafts(conn, "b")
    assert [(d["body"], d["fate"]) for d in drafts] == [
        ("kept one", "kept"),
        ("dropped one", "self_critique"),
    ]
    await record_verdicts(conn, [(drafts[0]["draft_id"], "keep", "real bug")])
    judged = await batch_drafts(conn, "b")
    assert (judged[0]["verdict"], judged[0]["verdict_reason"]) == ("keep", "real bug")
    assert judged[1]["verdict"] is None


async def test_a_silent_run_records_why(conn: asyncpg.Connection) -> None:
    run = await claim(conn, mode="replay", repo="o/r", pr_number=11, head_sha="h", batch="b")
    assert run is not None
    await finish(conn, run, Review(NoComment.RATE_LIMITED, detail="groq 429"), {})
    row = await conn.fetchrow("select no_comment, detail from review_runs where run_id = $1", run)
    assert row is not None
    assert (row["no_comment"], row["detail"]) == ("rate_limited", "groq 429")


async def test_a_verdict_is_keep_or_kill_and_nothing_else(conn: asyncpg.Connection) -> None:
    run = await claim(conn, mode="replay", repo="o/r", pr_number=12, head_sha="h", batch="b")
    assert run is not None
    await finish(conn, run, Review(None, drafts=[Draft(item(1, "x"), Fate.KEPT)]), {})
    [draft] = await batch_drafts(conn, "b")
    with pytest.raises(asyncpg.CheckViolationError):
        await record_verdicts(conn, [(draft["draft_id"], "maybe", "")])  # type: ignore[list-item]
