"""Exercised against the postgres service container in ci.yml, not against Neon.

Neon free is 0.5 GB with a CU-hour budget. Pointing a test suite at it burns the
production database's quota to prove code that a throwaway container proves for free.
"""

from collections.abc import AsyncIterator

import asyncpg
import pytest

from pr_lens.db.deliveries import DeliveryStore
from pr_lens.db.migrate import migrate
from pr_lens.models import Delivery

from .conftest import database_url, requires_postgres

pytestmark = requires_postgres


@pytest.fixture
async def pool() -> AsyncIterator[asyncpg.Pool]:
    dsn = database_url()
    assert dsn is not None
    await migrate(dsn)
    pool = await asyncpg.create_pool(dsn, min_size=1, max_size=2, statement_cache_size=0)
    async with pool.acquire() as conn:
        await conn.execute("truncate deliveries")
    yield pool
    await pool.close()


def delivery(delivery_id: str = "d-1", pr_number: int = 7) -> Delivery:
    return Delivery(
        delivery_id=delivery_id,
        event="pull_request",
        action="opened",
        repo_full_name="octocat/hello-world",
        pr_number=pr_number,
        head_sha="a" * 40,
        installation_id=12345,
    )


async def test_migrations_are_idempotent() -> None:
    dsn = database_url()
    assert dsn is not None
    await migrate(dsn)
    assert await migrate(dsn) == []


async def test_a_new_delivery_is_recorded(pool: asyncpg.Pool) -> None:
    store = DeliveryStore(pool)
    assert await store.record(delivery()) is True

    row = await pool.fetchrow("select * from deliveries where delivery_id = 'd-1'")
    assert row is not None
    assert row["repo_full_name"] == "octocat/hello-world"
    assert row["pr_number"] == 7
    assert row["received_at"] is not None
    assert row["dispatched_at"] is None


async def test_the_same_delivery_id_is_only_stored_once(pool: asyncpg.Pool) -> None:
    store = DeliveryStore(pool)
    assert await store.record(delivery()) is True
    # A redelivery of the same event carrying different data must not overwrite the row
    # or look like new work.
    assert await store.record(delivery(pr_number=99)) is False

    assert await pool.fetchval("select count(*) from deliveries") == 1
    assert await pool.fetchval("select pr_number from deliveries") == 7


async def test_marking_dispatched_records_the_outcome(pool: asyncpg.Pool) -> None:
    store = DeliveryStore(pool)
    await store.record(delivery())
    await store.mark_dispatched("d-1", "ok")

    row = await pool.fetchrow("select dispatched_at, dispatch_status from deliveries")
    assert row is not None
    assert row["dispatch_status"] == "ok"
    assert row["dispatched_at"] is not None


async def test_nullable_columns_accept_a_non_pull_request_event(pool: asyncpg.Pool) -> None:
    store = DeliveryStore(pool)
    ping = Delivery(
        delivery_id="d-ping",
        event="ping",
        action=None,
        repo_full_name=None,
        pr_number=None,
        head_sha=None,
        installation_id=None,
    )
    assert await store.record(ping) is True
