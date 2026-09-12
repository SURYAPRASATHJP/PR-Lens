"""Resolving a candidate list is one query, and that query never scans the table.

A sequential scan on corpus_units is a failure, not a slow success: the table is 153,758
rows in production, and a retriever that returns in milliseconds is worthless if the
lookup behind it reads the whole index. The plan is captured with enough rows loaded that
the planner has a real choice, because on a near-empty table it scans whatever you do.
"""

import json
from collections.abc import AsyncIterator
from typing import Any

import asyncpg
import pytest

from pr_lens.db.connection import connect
from pr_lens.db.corpus import RESOLVE_UNITS, resolve_units
from pr_lens.db.migrate import migrate

from .conftest import database_url, requires_postgres

pytestmark = requires_postgres

ROWS = 20_000


@pytest.fixture
async def conn() -> AsyncIterator[asyncpg.Connection]:
    dsn = database_url()
    assert dsn is not None
    await migrate(dsn)
    connection = await connect(dsn)
    await connection.execute("truncate corpus_units, ingest_runs, corpus_embeddings")
    await connection.copy_records_to_table(
        "corpus_units",
        records=[
            (f"unit-{i:06d}", f"owner/repo-{i % 18}", "source", f"src/m{i}.py", "h", 100, "s")
            for i in range(ROWS)
        ],
        columns=["unit_id", "repo", "kind", "path", "content_hash", "char_count", "shard"],
    )
    await connection.execute("analyze corpus_units")
    yield connection
    await connection.execute("truncate corpus_units, corpus_embeddings")
    await connection.close()


def node_types(plan: dict[str, Any]) -> list[str]:
    return [plan["Node Type"], *(t for child in plan.get("Plans", []) for t in node_types(child))]


async def test_thirty_candidates_resolve_without_a_sequential_scan(
    conn: asyncpg.Connection,
) -> None:
    ids = [f"unit-{i:06d}" for i in range(0, ROWS, ROWS // 30)][:30]
    raw = await conn.fetchval(f"explain (format json) {RESOLVE_UNITS}", ids)
    plan = json.loads(raw)[0]["Plan"]
    types = node_types(plan)
    assert "Seq Scan" not in types, types
    assert any("Index" in t for t in types), types


async def test_one_round_trip_returns_the_rows_in_retrieval_order(
    conn: asyncpg.Connection,
) -> None:
    wanted = ["unit-000042", "unit-019999", "missing-unit", "unit-000007"]
    rows = await resolve_units(conn, wanted)
    assert [row["unit_id"] for row in rows] == ["unit-000042", "unit-019999", "unit-000007"]
    assert rows[0]["path"] == "src/m42.py"
