"""One repo, end to end: fetch, chunk, shard, index.

The order matters. Shards are written before the rows that point at them, so a run killed
between the two leaves text in the dataset repo that nothing references, which is
recoverable. The reverse leaves rows citing shards that do not exist, which is not.
"""

import logging
from collections.abc import Sequence
from dataclasses import dataclass

import asyncpg

from pr_lens.corpus.writer import Sink, write_units
from pr_lens.db.corpus import UpsertCounts, finish_run, start_run, upsert_units
from pr_lens.github.client import GitHubClient
from pr_lens.ingest.mine import (
    MiningLimits,
    RepoHead,
    issue_units,
    pull_request_units,
    resolve_head,
    review_comment_units,
    source_and_doc_units,
)
from pr_lens.ingest.units import CorpusUnit

logger = logging.getLogger(__name__)

# Kinds where we see the complete set every run, so absence really does mean deleted.
# The API-backed kinds are windows over the most recent N, where absence only means
# "older than the window" and pruning would delete the corpus a page at a time.
COMPLETE_KINDS = ("source", "doc")


@dataclass(frozen=True, slots=True)
class IngestReport:
    repo: str
    ref: str
    counts: UpsertCounts
    rows: int
    shards: int
    shards_written: int
    pruned: int

    @property
    def wrote_nothing(self) -> bool:
        """The Phase 1 gate, for one repo."""
        return self.counts.wrote_nothing and self.shards_written == 0 and self.pruned == 0


async def ingest_repo(
    client: GitHubClient,
    conn: asyncpg.Connection,
    sink: Sink,
    repo: str,
    limits: MiningLimits,
) -> IngestReport:
    head = await resolve_head(client, repo)
    run_id = await start_run(conn, repo, head.sha)
    try:
        units = await _collect(client, head, limits)
        shard_report = write_units(sink, repo, units)
        counts = await upsert_units(conn, units, shard_report.placement)
        pruned = await prune_missing(conn, repo, units)
    except Exception as exc:
        await finish_run(
            conn, run_id, UpsertCounts(), shards=0, error=f"{type(exc).__name__}: {exc}"
        )
        raise

    await finish_run(conn, run_id, counts, shards=shard_report.shards)
    report = IngestReport(
        repo=repo,
        ref=head.sha,
        counts=counts,
        rows=shard_report.rows,
        shards=shard_report.shards,
        shards_written=len(shard_report.written),
        pruned=pruned,
    )
    logger.info(
        "%s at %s: %s units, %s inserted, %s updated, %s unchanged, %s pruned, "
        "%s/%s shards written",
        repo,
        head.sha[:7],
        counts.seen,
        counts.inserted,
        counts.updated,
        counts.unchanged,
        pruned,
        report.shards_written,
        report.shards,
    )
    return report


async def prune_missing(conn: asyncpg.Connection, repo: str, units: Sequence[CorpusUnit]) -> int:
    """Drop indexed source and doc units that are no longer in the tree.

    Without this, renaming a function leaves the old chunk indexed forever: retrieval
    keeps returning code that does not exist, and the citation points at a line that has
    moved. It is a delete rather than a soft flag because Neon free is 0.5 GB.
    """
    live = [(unit.unit_id,) for unit in units if unit.kind in COMPLETE_KINDS]
    if not live:
        # No tree was fetched this run, so we cannot tell deleted from not-looked-at.
        return 0

    async with conn.transaction():
        await conn.execute(
            "create temporary table live_units (unit_id text primary key) on commit drop"
        )
        await conn.copy_records_to_table("live_units", records=live, columns=["unit_id"])
        status = await conn.execute(
            """
            delete from corpus_units cu
            where cu.repo = $1
              and cu.kind = any($2::text[])
              and not exists (select 1 from live_units l where l.unit_id = cu.unit_id)
            """,
            repo,
            list(COMPLETE_KINDS),
        )
    return int(status.rsplit(" ", 1)[-1])


async def _collect(client: GitHubClient, head: RepoHead, limits: MiningLimits) -> list[CorpusUnit]:
    """Sequential on purpose.

    Running the four fetches concurrently would multiply the request rate against
    secondary limits that cannot be queried, to save wall-clock time on a job that runs
    overnight and has none to save.
    """
    return [
        *await source_and_doc_units(client, head),
        *await pull_request_units(client, head.repo, limits.pull_requests),
        *await review_comment_units(client, head.repo, limits.review_comments),
        *await issue_units(client, head.repo, limits.issues),
    ]
