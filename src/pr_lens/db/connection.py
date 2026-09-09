import logging
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

import asyncpg

logger = logging.getLogger(__name__)

# Neon hands out a URL carrying libpq options asyncpg does not accept. asyncpg forwards
# unknown query parameters to the server as settings, and Postgres then refuses the
# connection with "unrecognized configuration parameter". Strip them here rather than
# asking every caller to sanitise the string Neon gave them.
_UNSUPPORTED_QUERY_PARAMS = frozenset({"channel_binding", "pgbouncer", "options"})


def normalise_dsn(dsn: str) -> str:
    parts = urlsplit(dsn)
    params = parse_qsl(parts.query)
    dropped = {k for k, _ in params} & _UNSUPPORTED_QUERY_PARAMS
    if dropped:
        logger.debug("dropped dsn parameters asyncpg does not accept: %s", sorted(dropped))
    kept = [(k, v) for k, v in params if k not in _UNSUPPORTED_QUERY_PARAMS]
    return urlunsplit(parts._replace(query=urlencode(kept)))


async def connect(dsn: str) -> asyncpg.Connection:
    """A single connection, which is all an Actions job ever needs.

    statement_cache_size=0 is not a tuning choice. Neon's pooled endpoint is PgBouncer in
    transaction mode, where a statement prepared on one backend can be executed on
    another. asyncpg prepares by default, so leaving the cache on gives you a connection
    that works in testing and fails intermittently under concurrency.
    """
    return await asyncpg.connect(normalise_dsn(dsn), statement_cache_size=0)
