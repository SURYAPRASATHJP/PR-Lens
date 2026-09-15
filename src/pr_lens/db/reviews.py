import json
from collections.abc import Mapping, Sequence
from typing import Literal

import asyncpg

from pr_lens.review.pipeline import Review

Mode = Literal["live", "replay"]
Verdict = Literal["keep", "kill"]

_CLAIM = """
insert into review_runs (
    mode, repo, pr_number, head_sha, delivery_id, batch, source_repo, source_pr
) values ($1, $2, $3, $4, $5, $6, $7, $8)
on conflict do nothing
returning run_id
"""

_FINISH = """
update review_runs set
    finished_at = now(),
    no_comment = $2,
    detail = $3,
    plan = $4,
    hunks_shown = $5,
    hunks_dropped = $6,
    files_skipped = $7,
    past_shown = $8,
    prompt_tokens = $9,
    completion_tokens = $10,
    provider = $11,
    model = $12,
    timings = $13::jsonb
where run_id = $1
"""

_DRAFT = """
insert into drafts (
    run_id, position, path, line, body, evidence, critique,
    specific, non_obvious, grounded, fate, filter_reason
) values ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12)
"""

_BATCH_DRAFTS = """
select d.draft_id, d.path, d.line, d.body, d.evidence, d.critique, d.fate, d.filter_reason,
       d.verdict, d.verdict_reason, r.repo, r.pr_number, r.head_sha, r.plan
from drafts d join review_runs r using (run_id)
where r.batch = $1
order by r.repo, r.pr_number, d.position
"""


async def claim(
    conn: asyncpg.Connection,
    *,
    mode: Mode,
    repo: str,
    pr_number: int,
    head_sha: str,
    delivery_id: str | None = None,
    batch: str | None = None,
    source_repo: str | None = None,
    source_pr: int | None = None,
) -> int | None:
    """Take the run row before any tokens are spent. None means another job has it.

    The partial unique indexes decide, so two replay jobs on one pull request in one
    batch, or a re-run of a batch that half finished, draft each pull request once.
    """
    row = await conn.fetchrow(
        _CLAIM, mode, repo, pr_number, head_sha, delivery_id, batch, source_repo, source_pr
    )
    return None if row is None else int(row["run_id"])


async def finish(
    conn: asyncpg.Connection, run_id: int, review: Review, timings: Mapping[str, float]
) -> None:
    """The result and every draft with its fate, together or not at all."""
    first = review.calls[0] if review.calls else None
    async with conn.transaction():
        await conn.execute(
            _FINISH,
            run_id,
            review.no_comment.value if review.no_comment else None,
            review.detail,
            review.plan,
            len(review.shown),
            len(review.dropped),
            len(review.skipped),
            review.past_shown,
            sum(call.prompt_tokens for call in review.calls),
            sum(call.completion_tokens for call in review.calls),
            first.provider if first else None,
            first.model if first else None,
            json.dumps({stage: round(seconds, 3) for stage, seconds in timings.items()}),
        )
        await conn.executemany(
            _DRAFT,
            [
                (
                    run_id,
                    position,
                    draft.item.path,
                    draft.item.line,
                    draft.item.body,
                    draft.item.evidence,
                    draft.item.critique,
                    draft.item.specific,
                    draft.item.non_obvious,
                    draft.item.grounded,
                    draft.fate.value,
                    draft.filter_reason,
                )
                for position, draft in enumerate(review.drafts)
            ],
        )


async def batch_drafts(conn: asyncpg.Connection, batch: str) -> list[asyncpg.Record]:
    return list(await conn.fetch(_BATCH_DRAFTS, batch))


async def record_verdicts(
    conn: asyncpg.Connection, verdicts: Sequence[tuple[int, Verdict, str]]
) -> None:
    await conn.executemany(
        "update drafts set verdict = $2, verdict_reason = $3, verdict_at = now() "
        "where draft_id = $1",
        verdicts,
    )
