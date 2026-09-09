import asyncio
import logging
import os
import sys
from pathlib import Path

import asyncpg

from pr_lens.db.pool import normalise_dsn
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


def pending(applied: set[str]) -> list[Path]:
    return sorted(p for p in MIGRATIONS_DIR.glob("*.sql") if p.stem not in applied)


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
