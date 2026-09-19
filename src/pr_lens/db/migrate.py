import asyncio
import logging
import os
import sys
from pathlib import Path

import asyncpg

from pr_lens.db.connection import normalise_dsn
from pr_lens.logging import configure

logger = logging.getLogger(__name__)

MIGRATIONS_DIR = Path(__file__).parent / "migrations"

# Any 64-bit constant works. It only has to be the same in every process that migrates.
_ADVISORY_LOCK_KEY = 0x7052_4C45_4E53_0001

_BOOTSTRAP = """
create table if not exists schema_migrations (
    version    text primary key,
    applied_at timestamptz not null default now()
)
"""


class SchemaBehind(RuntimeError):
    """Migrations shipped with this code have not been applied to this database."""


def pending(applied: set[str]) -> list[Path]:
    return sorted(p for p in MIGRATIONS_DIR.glob("*.sql") if p.stem not in applied)


async def assert_current(conn: asyncpg.Connection) -> None:
    """Refuse to start a job whose schema is older than its code.

    Batch 2026-09-19-a-context spent a pull request's worth of inference and then died on
    `column "tool_turns" does not exist`, because migrations 0005 and 0006 shipped with the
    code and migrate.yml is run by hand. The first draft was paid for and thrown away, and
    the run row it left behind had to be released before the batch could be started again.

    A misconfigured job fails loudly here rather than going quiet. Silence is for a
    provider that said no, never for a deployment that was never finished.
    """
    try:
        rows = await conn.fetch("select version from schema_migrations")
    except asyncpg.UndefinedTableError as missing:
        raise SchemaBehind("this database has never been migrated") from missing
    missing_versions = [path.stem for path in pending({row["version"] for row in rows})]
    if missing_versions:
        raise SchemaBehind(
            f"not applied: {', '.join(missing_versions)}. Run the migrate workflow first."
        )


async def migrate(dsn: str) -> list[str]:
    """Apply every unapplied .sql file in filename order and return what ran.

    Each migration commits with its own version row, so a failure halfway through a set
    leaves the earlier ones applied and the failing one not. Re-running resumes.
    """
    conn = await asyncpg.connect(normalise_dsn(dsn), statement_cache_size=0)
    try:
        await conn.execute(_BOOTSTRAP)
        # Two Actions jobs starting at once would otherwise race on the same DDL.
        await conn.execute("select pg_advisory_lock($1)", _ADVISORY_LOCK_KEY)
        try:
            rows = await conn.fetch("select version from schema_migrations")
            applied = {r["version"] for r in rows}
            ran = []
            for path in pending(applied):
                logger.info("applying migration %s", path.stem)
                async with conn.transaction():
                    await conn.execute(path.read_text())
                    await conn.execute(
                        "insert into schema_migrations (version) values ($1)", path.stem
                    )
                ran.append(path.stem)
            return ran
        finally:
            await conn.execute("select pg_advisory_unlock($1)", _ADVISORY_LOCK_KEY)
    finally:
        await conn.close()


async def _main() -> None:
    """Read DATABASE_URL and nothing else.

    Migrating is an operations task. Requiring the webhook secret and the dispatch token
    to create a table would mean loading production credentials to run a schema change.
    """
    configure(os.environ.get("LOG_LEVEL", "INFO"))
    dsn = os.environ.get("DATABASE_URL")
    if not dsn:
        sys.exit("DATABASE_URL is not set")
    ran = await migrate(dsn)
    logger.info("migrations applied: %s", ", ".join(ran) if ran else "none, already current")


if __name__ == "__main__":
    asyncio.run(_main())
