import asyncpg
import pytest

from pr_lens.db.connection import normalise_dsn
from pr_lens.db.migrate import MIGRATIONS_DIR, SchemaBehind, assert_current


def test_drops_parameters_asyncpg_rejects() -> None:
    neon = (
        "postgresql://user:pw@ep-cool-name-pooler.us-east-2.aws.neon.tech/pr_lens"
        "?sslmode=require&channel_binding=require"
    )
    assert normalise_dsn(neon) == (
        "postgresql://user:pw@ep-cool-name-pooler.us-east-2.aws.neon.tech/pr_lens?sslmode=require"
    )


def test_keeps_sslmode_which_asyncpg_understands() -> None:
    assert "sslmode=require" in normalise_dsn("postgresql://h/db?sslmode=require")


def test_leaves_a_plain_local_dsn_alone() -> None:
    plain = "postgresql://postgres:postgres@localhost:5432/postgres"
    assert normalise_dsn(plain) == plain


class _FakeConn:
    """Just enough asyncpg.Connection for the schema guard."""

    def __init__(self, applied: list[str] | None) -> None:
        self._applied = applied

    async def fetch(self, query: str) -> list[dict[str, str]]:
        del query
        if self._applied is None:
            raise asyncpg.UndefinedTableError("relation does not exist")
        return [{"version": version} for version in self._applied]


async def test_a_job_whose_schema_is_older_than_its_code_refuses_to_start() -> None:
    """Batch 2026-09-19-a-context spent a pull request of inference and then died on
    `column "tool_turns" does not exist`, because migrations ship with the code and
    migrate.yml is run by hand. The draft was paid for and thrown away."""
    everything = [path.stem for path in sorted(MIGRATIONS_DIR.glob("*.sql"))]
    await assert_current(_FakeConn(everything))

    with pytest.raises(SchemaBehind) as behind:
        await assert_current(_FakeConn(everything[:-1]))
    assert everything[-1] in str(behind.value)

    with pytest.raises(SchemaBehind):
        await assert_current(_FakeConn(None))


async def test_the_guard_names_what_to_do_about_it() -> None:
    with pytest.raises(SchemaBehind) as behind:
        await assert_current(_FakeConn([]))
    assert "Run the migrate workflow first" in str(behind.value)
