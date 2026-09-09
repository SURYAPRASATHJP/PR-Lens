"""The Actions-side job. Exercised against the postgres service container in ci.yml.

Neon free is 0.5 GB with a 100 CU-hour budget; pointing a test suite at it spends the
production database's quota to prove what a throwaway container proves for nothing.
"""

import json
from collections.abc import AsyncIterator
from pathlib import Path

import asyncpg
import pytest

from pr_lens.db.connection import connect
from pr_lens.db.migrate import migrate
from pr_lens.jobs.record_delivery import emit_output, main, parse_payload

from .conftest import database_url, requires_postgres

pytestmark = requires_postgres

PAYLOAD = {
    "delivery_id": "d-1",
    "event": "pull_request",
    "action": "opened",
    "repo_full_name": "octocat/hello-world",
    "pr_number": 7,
    "head_sha": "a" * 40,
    "installation_id": 12345,
}


@pytest.fixture
async def conn() -> AsyncIterator[asyncpg.Connection]:
    dsn = database_url()
    assert dsn is not None
    await migrate(dsn)
    connection = await connect(dsn)
    await connection.execute("truncate deliveries")
    yield connection
    await connection.close()


@pytest.fixture(autouse=True)
def job_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    output = tmp_path / "github_output"
    output.touch()
    monkeypatch.setenv("DATABASE_URL", database_url() or "")
    monkeypatch.setenv("CLIENT_PAYLOAD", json.dumps(PAYLOAD))
    monkeypatch.setenv("GITHUB_OUTPUT", str(output))
    return output


async def test_a_new_delivery_is_written_and_reported_as_new(
    conn: asyncpg.Connection, job_env: Path
) -> None:
    assert await main() == 0

    assert "new=true" in job_env.read_text()
    row = await conn.fetchrow("select * from deliveries where delivery_id = 'd-1'")
    assert row is not None
    assert row["repo_full_name"] == "octocat/hello-world"
    assert row["pr_number"] == 7
    assert row["event"] == "pull_request"
    assert row["action"] == "opened"
    assert row["received_at"] is not None


async def test_a_redelivery_is_reported_as_not_new_and_writes_nothing(
    conn: asyncpg.Connection, job_env: Path
) -> None:
    assert await main() == 0
    job_env.write_text("")

    assert await main() == 0

    assert "new=false" in job_env.read_text()
    assert await conn.fetchval("select count(*) from deliveries") == 1


async def test_a_redelivery_carrying_different_data_does_not_overwrite_the_row(
    conn: asyncpg.Connection, job_env: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    await main()
    monkeypatch.setenv("CLIENT_PAYLOAD", json.dumps({**PAYLOAD, "pr_number": 99}))

    assert await main() == 0
    assert await conn.fetchval("select pr_number from deliveries") == 7


async def test_a_missing_database_url_fails_loudly(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("DATABASE_URL")
    assert await main() == 1


async def test_a_missing_payload_fails_loudly(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("CLIENT_PAYLOAD")
    assert await main() == 1


async def test_a_payload_that_is_not_the_shape_we_send_fails_loudly(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CLIENT_PAYLOAD", json.dumps({"delivery_id": "d-1"}))
    assert await main() == 1


async def test_a_non_json_payload_fails_loudly(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CLIENT_PAYLOAD", "not json")
    assert await main() == 1


def test_parse_payload_rebuilds_what_the_receiver_sent() -> None:
    delivery = parse_payload(json.dumps(PAYLOAD))
    assert delivery.delivery_id == "d-1"
    assert delivery.pr_number == 7
    assert delivery.as_client_payload() == PAYLOAD


def test_emit_output_falls_back_to_a_log_line_without_github_output(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("GITHUB_OUTPUT")
    emit_output("new", "true")  # must not raise when run outside Actions
