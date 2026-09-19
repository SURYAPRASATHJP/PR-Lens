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
    timings = $13::jsonb,
    reference = $14::jsonb,
    head_sha = coalesce(nullif($15, ''), head_sha),
    verification = $16::jsonb,
    tool_turns = $17,
    tools_used = $18
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
       d.verdict, d.verdict_reason, r.run_id, r.repo, r.pr_number, r.head_sha, r.plan,
       r.reference
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
    conn: asyncpg.Connection,
    run_id: int,
    review: Review,
    timings: Mapping[str, float],
    reference: Sequence[Mapping[str, object]] = (),
    *,
    head_sha: str = "",
) -> None:
    """The result and every draft with its fate, together or not at all. A head_sha given
    here replaces the one claimed with, since replay claims before it has read the pull."""
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
            json.dumps(list(reference)),
            head_sha,
            json.dumps(_verification(review)),
            review.grounding.turns or None,
            review.grounding.summary or None,
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


def _verification(review: Review) -> dict[str, object]:
    """The sandbox's answer, small enough to keep beside the run."""
    checked = review.verification
    if checked is None:
        return {}
    return {
        "ran": checked.ran,
        "reason": checked.reason,
        "selected": len(checked.selected),
        "regressions": len(checked.regressions),
        "head_outcome": checked.head_outcome,
        "base_outcome": checked.base_outcome,
        "seconds": checked.seconds,
    }


async def release(conn: asyncpg.Connection, run_id: int) -> None:
    """Give back a claim that spent nothing, so a later run can take the pull request."""
    await conn.execute("delete from review_runs where run_id = $1 and finished_at is null", run_id)


async def batch_drafts(conn: asyncpg.Connection, batch: str) -> list[asyncpg.Record]:
    return list(await conn.fetch(_BATCH_DRAFTS, batch))


async def silent_runs(conn: asyncpg.Connection, batch: str) -> list[asyncpg.Record]:
    """The batch's pull requests that produced no draft at all, with why.

    A rate limit must not read as nothing to report. Without these rows a pull request the
    provider refused looks exactly like one nobody drafted anything for.
    """
    return list(
        await conn.fetch(
            "select repo, pr_number, no_comment, detail, verification from review_runs "
            "where batch = $1 and not exists (select 1 from drafts where drafts.run_id = "
            "review_runs.run_id) order by repo, pr_number",
            batch,
        )
    )


async def replayed_elsewhere(
    conn: asyncpg.Connection, batch: str, compare: str | None = None
) -> set[tuple[str, int]]:
    """Every pull request another replay batch has drafted, so a new batch reads new ones.

    Not this batch's own: a re-run of a half-finished batch has to choose the same pull
    requests again, and the claim is what skips the ones already drafted.

    Not `compare`'s either. Measuring a change to the reviewer means drafting the same
    pull requests again with the new code, and the ordinary rule makes that impossible:
    a new batch would skip every pull request the old one read, which is all of them.
    Phase 4b is context, measure, tools, measure, verifier, and each of those measures is
    this comparison, so the harness has to allow it.
    """
    rows = await conn.fetch(
        "select repo, pr_number from review_runs "
        "where mode = 'replay' and batch <> $1 and ($2::text is null or batch <> $2)",
        batch,
        compare,
    )
    return {(row["repo"], row["pr_number"]) for row in rows}


async def record_verdicts(
    conn: asyncpg.Connection, verdicts: Sequence[tuple[int, Verdict, str]]
) -> None:
    await conn.executemany(
        "update drafts set verdict = $2, verdict_reason = $3, verdict_at = now() "
        "where draft_id = $1",
        verdicts,
    )
