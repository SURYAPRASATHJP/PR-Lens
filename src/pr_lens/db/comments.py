import asyncpg

_TAKE = """
insert into posted_comments (repo, pr_number, slot, path, line, run_id, body_sha256)
values ($1, $2, $3, $4, $5, $6, $7)
on conflict do nothing
returning slot
"""

_LINE_TAKEN = """
select 1 from posted_comments where repo = $1 and pr_number = $2 and path = $3 and line = $4
"""


async def take_slot(
    conn: asyncpg.Connection,
    *,
    repo: str,
    pr_number: int,
    path: str,
    line: int,
    run_id: int | None,
    body_sha256: str,
    cap: int,
) -> int | None:
    """The first free slot for this comment, or None if it must not be posted.

    None for either of the two reasons the constraints give: the line already carries a
    PR-Lens comment, or every slot up to the cap is taken. Each insert is its own
    statement, so two runs racing for one slot cannot both win it.
    """
    for slot in range(1, cap + 1):
        row = await conn.fetchrow(_TAKE, repo, pr_number, slot, path, line, run_id, body_sha256)
        if row is not None:
            return int(row["slot"])
        if await conn.fetchval(_LINE_TAKEN, repo, pr_number, path, line):
            return None
    return None


async def mark_posted(
    conn: asyncpg.Connection, repo: str, pr_number: int, slot: int, comment_id: int
) -> None:
    await conn.execute(
        "update posted_comments set status = 'posted', github_comment_id = $4, "
        "posted_at = now() where repo = $1 and pr_number = $2 and slot = $3",
        repo,
        pr_number,
        slot,
        comment_id,
    )


async def mark_failed(
    conn: asyncpg.Connection, repo: str, pr_number: int, slot: int, error: str
) -> None:
    await conn.execute(
        "update posted_comments set status = 'failed', error = $4 "
        "where repo = $1 and pr_number = $2 and slot = $3",
        repo,
        pr_number,
        slot,
        error[:500],
    )
