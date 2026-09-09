import logging
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

import asyncpg

logger = logging.getLogger(__name__)

# Neon hands out a URL carrying libpq options that asyncpg does not accept. It passes
# unknown query parameters through to the server as settings, and Postgres then rejects
# the connection with "unrecognized configuration parameter". Drop them here rather than
# asking every caller to sanitise the string they were given.
_UNSUPPORTED_QUERY_PARAMS = frozenset({"channel_binding", "pgbouncer", "options"})


def normalise_dsn(dsn: str) -> str:
    parts = urlsplit(dsn)
    kept = [(k, v) for k, v in parse_qsl(parts.query) if k not in _UNSUPPORTED_QUERY_PARAMS]
    dropped = {k for k, _ in parse_qsl(parts.query)} & _UNSUPPORTED_QUERY_PARAMS
    if dropped:
        logger.debug("dropped dsn parameters asyncpg does not accept: %s", sorted(dropped))
    return urlunsplit(parts._replace(query=urlencode(kept)))


async def create_pool(dsn: str) -> asyncpg.Pool:
    """Open a pool sized for a Space that is idle most of the time.

    statement_cache_size=0 is not a tuning choice. Neon's pooled endpoint is PgBouncer in
    transaction mode, where a prepared statement can be issued on one backend and executed
    on another. asyncpg prepares by default, so leaving the cache on gives you connections
    that work in testing and fail intermittently under concurrency.
    """
    return await asyncpg.create_pool(
        normalise_dsn(dsn),
        min_size=0,
        max_size=4,
        statement_cache_size=0,
        command_timeout=10.0,
    )
