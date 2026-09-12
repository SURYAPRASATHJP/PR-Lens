"""Idempotent writes to the corpus index.

The Phase 1 gate is that re-running ingest writes zero new rows, so the behaviour this
file has to get exactly right is doing nothing. A unit whose content hash matches what is
already stored produces no write at all, not a no-op update: an update would still churn
updated_at, still spend Neon's write budget, and still make the gate unmeasurable.
"""

import json
import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import asyncpg

from pr_lens.ingest.units import CorpusUnit

logger = logging.getLogger(__name__)

# Bounded so that a repo with a hundred thousand chunks does not build one enormous
# statement. Large enough that the round trips are not the bottleneck.
BATCH_SIZE = 1000

_UPSERT = """
insert into corpus_units (
    unit_id, repo, kind, ref, path, symbol,
    start_line, end_line, content_hash, char_count, shard, metadata
)
select
    unit_id, repo, kind, ref, path, symbol,
    start_line, end_line, content_hash, char_count, shard, metadata::jsonb
from unnest(
    $1::text[], $2::text[], $3::text[], $4::text[], $5::text[], $6::text[],
    $7::integer[], $8::integer[], $9::text[], $10::integer[], $11::text[], $12::text[]
) as incoming (
    unit_id, repo, kind, ref, path, symbol,
    start_line, end_line, content_hash, char_count, shard, metadata
)
on conflict (unit_id) do update set
    repo = excluded.repo,
    kind = excluded.kind,
    ref = excluded.ref,
    path = excluded.path,
    symbol = excluded.symbol,
    start_line = excluded.start_line,
    end_line = excluded.end_line,
    content_hash = excluded.content_hash,
    char_count = excluded.char_count,
    shard = excluded.shard,
    metadata = excluded.metadata,
    updated_at = now()
where corpus_units.content_hash is distinct from excluded.content_hash
returning unit_id, (xmax = 0) as inserted
"""


@dataclass(frozen=True, slots=True)
class UpsertCounts:
    inserted: int = 0
    updated: int = 0
    unchanged: int = 0

    @property
    def seen(self) -> int:
        return self.inserted + self.updated + self.unchanged

    @property
    def wrote_nothing(self) -> bool:
        """What the gate asserts on a second pass."""
        return self.inserted == 0 and self.updated == 0

    def __add__(self, other: "UpsertCounts") -> "UpsertCounts":
        return UpsertCounts(
            inserted=self.inserted + other.inserted,
            updated=self.updated + other.updated,
            unchanged=self.unchanged + other.unchanged,
        )


async def upsert_units(
    conn: asyncpg.Connection,
    units: Sequence[CorpusUnit],
    placement: Mapping[str, str],
) -> UpsertCounts:
    """Write the units whose content changed and count the ones that did not."""
    total = UpsertCounts()
    for batch in _batched(_deduplicate(units)):
        rows = await conn.fetch(_UPSERT, *_columns(batch, placement))
        inserted = sum(1 for row in rows if row["inserted"])
        total += UpsertCounts(
            inserted=inserted,
            updated=len(rows) - inserted,
            unchanged=len(batch) - len(rows),
        )
    return total


async def start_run(conn: asyncpg.Connection, repo: str, ref: str | None) -> int:
    row = await conn.fetchrow(
        "insert into ingest_runs (repo, ref) values ($1, $2) returning run_id", repo, ref
    )
    assert row is not None
    return int(row["run_id"])


async def finish_run(
    conn: asyncpg.Connection,
    run_id: int,
    counts: UpsertCounts,
    *,
    shards: int,
    error: str | None = None,
) -> None:
    await conn.execute(
        """
        update ingest_runs set
            finished_at = now(),
            units_seen = $2, inserted = $3, updated = $4, unchanged = $5,
            shards = $6, error = $7
        where run_id = $1
        """,
        run_id,
        counts.seen,
        counts.inserted,
        counts.updated,
        counts.unchanged,
        shards,
        error,
    )


def _deduplicate(units: Sequence[CorpusUnit]) -> list[CorpusUnit]:
    """Postgres refuses an upsert that touches the same row twice in one statement.

    A duplicate here means two units resolved to the same identity, which is a bug in
    whichever ingest path produced them rather than something to paper over quietly.
    """
    seen: dict[str, CorpusUnit] = {}
    for unit in units:
        if unit.unit_id in seen:
            logger.warning(
                "two units share the identity %s|%s|%s, keeping the first",
                unit.repo,
                unit.kind,
                unit.identity,
            )
            continue
        seen[unit.unit_id] = unit
    return list(seen.values())


def _batched(units: Sequence[CorpusUnit]) -> list[Sequence[CorpusUnit]]:
    return [units[i : i + BATCH_SIZE] for i in range(0, len(units), BATCH_SIZE)]


def _columns(batch: Sequence[CorpusUnit], placement: Mapping[str, str]) -> list[list[object]]:
    """The upsert takes one array per column, so the batch is transposed to send it."""
    rows = [unit.as_row() for unit in batch]
    return [
        [r["unit_id"] for r in rows],
        [r["repo"] for r in rows],
        [r["kind"] for r in rows],
        [r["ref"] for r in rows],
        [r["path"] for r in rows],
        [r["symbol"] for r in rows],
        [r["start_line"] for r in rows],
        [r["end_line"] for r in rows],
        [r["content_hash"] for r in rows],
        [r["char_count"] for r in rows],
        [placement.get(str(r["unit_id"]), "") for r in rows],
        [json.dumps(r["metadata"], sort_keys=True) for r in rows],
    ]


# One round trip for a whole candidate list. The obvious version, one lookup per retrieved
# unit, is thirty round trips to a database that may be waking from scale-to-zero, and it
# is the N+1 that makes a fast retriever look slow.
RESOLVE_UNITS = """
select unit_id, repo, kind, path, symbol, start_line, end_line, shard
from corpus_units
where unit_id = any($1::text[])
"""


async def resolve_units(conn: asyncpg.Connection, unit_ids: Sequence[str]) -> list[asyncpg.Record]:
    """Where each retrieved unit lives, in retrieval order, for citing and fetching text."""
    rows = await conn.fetch(RESOLVE_UNITS, list(unit_ids))
    by_id = {row["unit_id"]: row for row in rows}
    return [by_id[unit_id] for unit_id in unit_ids if unit_id in by_id]
