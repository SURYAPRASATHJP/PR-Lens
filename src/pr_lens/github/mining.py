"""What every job that reads GitHub for mining shares: where it caches, and as whom.

Kept out of jobs/ingest.py on purpose. That module opens Postgres, so importing even a
constant from it pulls asyncpg into jobs whose runners never install the database driver,
and they die at import. The first Phase 2 pairs run did exactly that in 19 seconds.
tests/test_job_dependencies.py imports every database-free job in a clean interpreter and
fails if the driver comes along.
"""

import os
from pathlib import Path

DEFAULT_CACHE_DIR = Path(".cache/github")


def mining_token() -> str | None:
    """GH_MINING_TOKEN if the mining run has its own credential, else the dispatch PAT.

    Never the workflow's GITHUB_TOKEN: it is capped at 1,000 requests per hour per
    repository, and the client refuses it rather than running at a fifth of the speed and
    looking like a slow network.
    """
    return os.environ.get("GH_MINING_TOKEN") or os.environ.get("GH_DISPATCH_TOKEN")
